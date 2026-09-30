"""Local demo UI for the Day 14 evaluation lab.

Serves ``demo/static`` and a small JSON API on top of the existing lab code:
``domain_assistant.py`` (system under evaluation) and ``template.py``
(evaluation core). No dependency outside ``requirements.txt`` is used.

Run from the repository root:
    python demo/server.py            # http://127.0.0.1:8014
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Private helpers are imported on purpose: the demo must reuse the exact
# tokenization / prompt / metric code that the benchmark uses.
from domain_assistant import (  # noqa: E402
    STOPWORDS as RETRIEVER_STOPWORDS,
    TOKEN_RE,
    BM25Retriever,
    Chunk,
    OpenAIGenerator,
    TextGenerator,
    _build_prompt,
    _normalize,
    _tokenize as retriever_tokenize,
    generate_actual_answers,
    load_corpus,
)
from evaluate_answers import build_evaluation_artifact, load_evaluation_inputs  # noqa: E402
from template import (  # noqa: E402
    BenchmarkRunner,
    EvalResult,
    FailureAnalyzer,
    RAGASEvaluator,
    _coverage,
    _tokenize as metric_tokenize,
    rerank_by_overlap,
)

CORPUS_DIR = ROOT / "data" / "technology_store"
GOLDEN_PATH = ROOT / "golden_dataset.json"
SEED_ANSWERS = ROOT / "artifacts" / "actual_answers.json"
RUNS_DIR = ROOT / "artifacts" / "runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
NOTES_PATH = Path(__file__).resolve().parent / "case_notes.json"

RUN_ID_RE = re.compile(r"^run-\d{3}$")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
RELEVANCE_THRESHOLD = 0.1  # same default as evaluate_context_precision
REGRESSION_THRESHOLD = BenchmarkRunner.REGRESSION_THRESHOLD
ANSWER_METRICS = ("faithfulness", "relevance", "completeness")
REVIEW_DIFFICULTIES = {"hard", "adversarial"}

# Experimental prompt addendum (the fix proposed in reflection.md). It is
# injected by wrapping the generator, so domain_assistant.py stays untouched
# and the baseline remains reproducible.
PROMPT_VARIANTS: dict[str, str] = {
    "1.0": "",
    "1.1": (
        "Additional rules:\n"
        "- If the request is unrelated to OrbitTech customer support, briefly say "
        "you only handle OrbitTech support and list supported topics (products, "
        "orders, payments, promotions, shipping, returns, warranty, repairs, "
        "accounts).\n"
        "- When a policy depends on a date, first identify the order or event "
        "date, decide which policy version applies, and apply only that "
        "version's rules.\n"
        "- When a rule says 'the longer of X or Y', compute both values and state "
        "which one applies.\n"
        "- End with one clear final conclusion that does not contradict earlier "
        "sentences."
    ),
}


class VariantGenerator:
    """Wraps a generator and inserts extra prompt rules before the question."""

    def __init__(self, base: TextGenerator, addendum: str) -> None:
        self.base = base
        self.addendum = addendum
        self.model = getattr(base, "model", base.__class__.__name__)

    def generate(self, prompt: str) -> str:
        if self.addendum:
            prompt = prompt.replace(
                "\nQuestion:\n", f"\n{self.addendum}\n\nQuestion:\n", 1
            )
        return self.base.generate(prompt)


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

CORPUS_ID, CHUNKS = load_corpus(CORPUS_DIR)
CHUNKS_BY_ID: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in CHUNKS}
RETRIEVER = BM25Retriever(CHUNKS)
EVALUATOR = RAGASEvaluator()
RUNNER = BenchmarkRunner()
ANALYZER = FailureAnalyzer()
RUN_LOCK = threading.Lock()
_generator: OpenAIGenerator | None = None


def _load_golden() -> dict[str, dict[str, Any]]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return {pair["id"]: pair for pair in data["qa_pairs"]}


GOLDEN = _load_golden()


def _load_notes() -> dict[str, Any]:
    if not NOTES_PATH.exists():
        return {}
    return json.loads(NOTES_PATH.read_text(encoding="utf-8"))


NOTES = _load_notes()


def has_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and key != "your_openai_api_key_here"


def get_generator() -> OpenAIGenerator:
    global _generator
    if not has_api_key():
        raise ApiError("OPENAI_API_KEY chưa được cấu hình trong .env", HTTPStatus.CONFLICT)
    if _generator is None:
        _generator = OpenAIGenerator()
    return _generator


# ---------------------------------------------------------------------------
# Runs (saved answer artifacts + their evaluation)
# ---------------------------------------------------------------------------

def _run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id or ""):
        raise ApiError(f"run id không hợp lệ: {run_id!r}")
    path = RUNS_DIR / run_id
    if not path.is_dir():
        raise ApiError(f"Không tìm thấy run {run_id}", HTTPStatus.NOT_FOUND)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """Evaluate a run's saved answers with the lab core, exactly like evaluate_answers.py."""
    results = _eval_results(run_dir)
    summary = RUNNER.generate_report(results)
    artifact = build_evaluation_artifact(results, summary, ANALYZER)
    _write_json(run_dir / "benchmark_results.json", artifact)
    return artifact


def _eval_results(run_dir: Path) -> list[EvalResult]:
    qa_pairs, answers = load_evaluation_inputs(GOLDEN_PATH, run_dir / "actual_answers.json")
    return RUNNER.run(qa_pairs, lambda question: answers[question], EVALUATOR)


def _next_run_id() -> str:
    numbers = [
        int(path.name.split("-")[1])
        for path in RUNS_DIR.glob("run-*")
        if RUN_ID_RE.match(path.name)
    ]
    return f"run-{max(numbers, default=0) + 1:03d}"


def _save_run(answers_artifact: dict[str, Any], label: str, source: str) -> dict[str, Any]:
    run_id = _next_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "actual_answers.json", answers_artifact)
    agent = answers_artifact.get("agent", {})
    meta = {
        "id": run_id,
        "label": label,
        "source": source,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generated_at": answers_artifact.get("generated_at"),
        "model": agent.get("model"),
        "top_k": agent.get("top_k"),
        "prompt_version": agent.get("prompt_version"),
    }
    _write_json(run_dir / "meta.json", meta)
    evaluate_run(run_dir)
    return meta


def seed_runs() -> None:
    """Register the committed benchmark artifact as the first (baseline) run."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if any(RUNS_DIR.glob("run-*")) or not SEED_ANSWERS.exists():
        return
    artifact = json.loads(SEED_ANSWERS.read_text(encoding="utf-8"))
    _save_run(artifact, "Baseline — artifacts/actual_answers.json", "seed")


def list_runs() -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(RUNS_DIR.glob("run-*")):
        meta_path = run_dir / "meta.json"
        bench_path = run_dir / "benchmark_results.json"
        if not (meta_path.exists() and bench_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["summary"] = json.loads(bench_path.read_text(encoding="utf-8"))["summary"]
        runs.append(meta)
    return runs


def load_run(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    answers = json.loads((run_dir / "actual_answers.json").read_text(encoding="utf-8"))
    bench = json.loads((run_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "meta": meta,
        "summary": bench["summary"],
        "answers": {record["id"]: record for record in answers["answers"]},
        "results": {record["id"]: record for record in bench["results"]},
    }


def create_live_run(prompt_version: str, top_k: int, label: str) -> dict[str, Any]:
    if prompt_version not in PROMPT_VARIANTS:
        raise ApiError(f"prompt_version không hỗ trợ: {prompt_version}")
    generator = VariantGenerator(get_generator(), PROMPT_VARIANTS[prompt_version])
    with RUN_LOCK:
        artifact = generate_actual_answers(GOLDEN_PATH, CORPUS_DIR, generator, top_k)
        artifact["agent"]["prompt_version"] = prompt_version
        return _save_run(artifact, label or f"Live run — prompt {prompt_version}", "live")


# ---------------------------------------------------------------------------
# Retrieval, generation and case diagnostics
# ---------------------------------------------------------------------------

def _chunk_record(chunk: Chunk, bm25_rank: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_doc": chunk.source_doc,
        "text": chunk.text,
        "score": round(chunk.score, 6),
        "bm25_rank": bm25_rank,
    }


def retrieve(question: str, top_k: int, rerank: bool) -> list[dict[str, Any]]:
    chunks = RETRIEVER.retrieve(question, top_k)
    records = [_chunk_record(chunk, rank) for rank, chunk in enumerate(chunks, start=1)]
    if rerank:
        order = rerank_by_overlap([record["text"] for record in records], question)
        remaining = list(records)
        reordered = []
        for text in order:
            index = next(i for i, record in enumerate(remaining) if record["text"] == text)
            reordered.append(remaining.pop(index))
        records = reordered
    return records


def generate_answer(question: str, records: list[dict[str, Any]], prompt_version: str) -> str:
    if prompt_version not in PROMPT_VARIANTS:
        raise ApiError(f"prompt_version không hỗ trợ: {prompt_version}")
    chunks = [
        CHUNKS_BY_ID[record["chunk_id"]]
        for record in records
        if record["chunk_id"] in CHUNKS_BY_ID
    ]
    generator = VariantGenerator(get_generator(), PROMPT_VARIANTS[prompt_version])
    answer = generator.generate(_build_prompt(question, chunks)).strip()
    if not answer:
        raise ApiError("Model trả về câu trả lời rỗng", HTTPStatus.BAD_GATEWAY)
    return answer


def _lexical_match(question: str, chunk_id: str, text: str) -> dict[str, Any]:
    """Explain the BM25 match: which normalized query terms the chunk shares."""
    query_terms = set(retriever_tokenize(question))
    chunk = CHUNKS_BY_ID.get(chunk_id)
    title = chunk.title if chunk else ""
    chunk_terms = set(retriever_tokenize(f"{title} {text}"))
    matched = sorted(query_terms & chunk_terms)
    highlight = sorted(
        {
            word
            for word in TOKEN_RE.findall(text.lower())
            if word not in RETRIEVER_STOPWORDS and _normalize(word) in matched
        }
    )
    return {"matched_terms": matched, "highlight": highlight}


def describe_chunks(
    question: str,
    records: list[dict[str, Any]],
    reference: str | None,
    gold_texts: list[str],
) -> list[dict[str, Any]]:
    reference_tokens = metric_tokenize(reference or "")
    described = []
    for rank, record in enumerate(records, start=1):
        text = record["text"]
        item = {
            "rank": rank,
            "bm25_rank": record.get("bm25_rank", rank),
            "chunk_id": record.get("chunk_id"),
            "source_doc": record.get("source_doc"),
            "score": record.get("score"),
            "text": text,
            "gold_evidence": [evidence for evidence in gold_texts if evidence in text],
            **_lexical_match(question, record.get("chunk_id", ""), text),
        }
        if reference:
            coverage = _coverage(reference_tokens, metric_tokenize(text))
            item["coverage"] = round(coverage, 3)
            item["relevant"] = coverage >= RELEVANCE_THRESHOLD
        described.append(item)
    return described


def _numbers(text: str) -> set[str]:
    """Figures worth checking; bare single digits ("September 1") are too noisy."""
    values = {match.replace(",", "") for match in NUMBER_RE.findall(text or "")}
    return {value for value in values if len(value) > 1}


def number_check(
    question: str, answer: str, expected: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare figures (days, USD, %) — the thing word overlap cannot judge."""
    in_question = _numbers(question)
    in_answer = _numbers(answer)
    in_expected = _numbers(expected)
    chunk_numbers = [(chunk["rank"], _numbers(chunk["text"])) for chunk in chunks]
    missing = [
        {
            "value": value,
            "ranks": [rank for rank, values in chunk_numbers if value in values],
        }
        for value in sorted(in_expected - in_answer - in_question, key=_num_key)
    ]
    unsupported = [
        value
        for value in sorted(in_answer - in_question, key=_num_key)
        if value not in in_expected and not any(value in values for _, values in chunk_numbers)
    ]
    return {"missing": missing, "unsupported": unsupported}


def _num_key(value: str) -> float:
    try:
        return float(value.rstrip("%"))
    except ValueError:
        return 0.0


def _scores(result: EvalResult) -> dict[str, Any]:
    return {
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "completeness": result.completeness,
        "context_recall": result.context_recall,
        "context_precision": result.context_precision,
        "overall": result.overall_score(),
        "passed": result.passed,
        "failure_type": result.failure_type,
        "root_cause": None if result.passed else ANALYZER.find_root_cause(result),
    }


def build_case_view(
    question: str,
    answer: str | None,
    records: list[dict[str, Any]],
    case_id: str | None,
    reference: str | None,
) -> dict[str, Any]:
    """Assemble answer + chunks + (only when grounded) evaluation for the UI."""
    pair = GOLDEN.get(case_id) if case_id else None
    gold_texts = [context["text"].strip() for context in pair["contexts"]] if pair else []
    expected = pair["expected_answer"] if pair else (reference or "").strip() or None
    chunks = describe_chunks(question, records, expected, gold_texts)
    texts = [record["text"] for record in records]

    view: dict[str, Any] = {
        "case_id": case_id,
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "expected_answer": expected,
        "difficulty": pair.get("difficulty") if pair else None,
        "attack_type": pair.get("attack_type") if pair else None,
        "evaluation": None,
        "eval_basis": None,
        "query_terms": _query_term_map(question),
    }

    if expected and texts:
        view["retrieval"] = {
            "context_recall": EVALUATOR.evaluate_context_recall(texts, expected),
            "context_precision": EVALUATOR.evaluate_context_precision(texts, expected),
        }

    if answer is None or expected is None:
        return view

    if pair:
        # Same inputs as the benchmark: faithfulness against gold evidence.
        result = EVALUATOR.run_full_eval(
            answer, question, "\n\n".join(gold_texts), expected, contexts=texts
        )
        view["eval_basis"] = "golden"
        view["gold_evidence"] = [
            {
                "source_doc": context["source_doc"],
                "text": context["text"].strip(),
                "rank": next(
                    (chunk["rank"] for chunk in chunks if context["text"].strip() in chunk["text"]),
                    None,
                ),
            }
            for context in pair["contexts"]
        ]
        view["note"] = NOTES.get(case_id)
    else:
        # User-supplied reference: no gold evidence exists, so faithfulness is
        # measured against the retrieved chunks and labelled as such in the UI.
        result = EVALUATOR.run_full_eval(
            answer, question, "\n\n".join(texts), expected, contexts=texts
        )
        view["eval_basis"] = "user_reference"

    view["evaluation"] = _scores(result)
    view["numbers"] = number_check(question, answer, expected, chunks)
    return view


def _query_term_map(question: str) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for word in TOKEN_RE.findall(question.lower()):
        if word in RETRIEVER_STOPWORDS:
            continue
        seen.setdefault(word, _normalize(word))
    return [{"raw": raw, "norm": norm} for raw, norm in seen.items()]


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

def _int(value: Any, default: int, low: int = 1, high: int = 10) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def api_status(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_key": has_api_key(),
        "model": os.getenv("OPENAI_MODEL", "").strip() or None,
        "corpus_id": CORPUS_ID,
        "chunks": len(CHUNKS),
        "prompt_versions": list(PROMPT_VARIANTS),
    }


def api_cases(_: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": pair["id"],
            "difficulty": pair["difficulty"],
            "attack_type": pair["attack_type"],
            "question": pair["question"],
            "expected_answer": pair["expected_answer"],
            "has_note": pair["id"] in NOTES,
        }
        for pair in GOLDEN.values()
    ]


def api_runs(_: dict[str, Any]) -> list[dict[str, Any]]:
    return list_runs()


def api_run(params: dict[str, Any]) -> dict[str, Any]:
    run = load_run(params.get("id", ""))
    return {"meta": run["meta"], "summary": run["summary"], "results": list(run["results"].values())}


def api_case(params: dict[str, Any]) -> dict[str, Any]:
    """Saved answer + trace for one golden case in one run."""
    case_id = params.get("case", "")
    if case_id not in GOLDEN:
        raise ApiError(f"Không có case {case_id}", HTTPStatus.NOT_FOUND)
    run = load_run(params.get("run", ""))
    record = run["answers"].get(case_id)
    if record is None:
        raise ApiError(f"Run không có câu trả lời cho {case_id}", HTTPStatus.NOT_FOUND)
    records = [
        {**context, "bm25_rank": rank}
        for rank, context in enumerate(record["retrieved_contexts"], start=1)
    ]
    view = build_case_view(record["question"], record["actual_answer"], records, case_id, None)
    view["source"] = {"kind": "saved", "run_id": run["meta"]["id"], **_run_config(run["meta"])}
    return view


def _run_config(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": meta.get("model"),
        "top_k": meta.get("top_k"),
        "prompt_version": meta.get("prompt_version"),
        "rerank": False,
    }


def api_ask(body: dict[str, Any]) -> dict[str, Any]:
    """Live retrieval (+ generation when a key exists) for a golden or new question."""
    case_id = body.get("case_id") or None
    if case_id is not None and case_id not in GOLDEN:
        raise ApiError(f"Không có case {case_id}")
    question = GOLDEN[case_id]["question"] if case_id else str(body.get("question", "")).strip()
    if not question:
        raise ApiError("Hãy nhập câu hỏi")
    top_k = _int(body.get("top_k"), 5)
    rerank = bool(body.get("rerank"))
    prompt_version = str(body.get("prompt_version", "1.0"))
    reference = None if case_id else (str(body.get("reference") or "").strip() or None)

    records = retrieve(question, top_k, rerank)
    answer: str | None = None
    warning: str | None = None
    if body.get("generate", True):
        if has_api_key():
            answer = generate_answer(question, records, prompt_version)
        else:
            warning = "Không có OPENAI_API_KEY — chỉ hiển thị kết quả retrieval."

    view = build_case_view(question, answer, records, case_id, reference)
    view["warning"] = warning
    view["source"] = {
        "kind": "live",
        "model": getattr(_generator, "model", None) or os.getenv("OPENAI_MODEL"),
        "top_k": top_k,
        "rerank": rerank,
        "prompt_version": prompt_version,
    }
    return view


def api_retrieve(body: dict[str, Any]) -> dict[str, Any]:
    return api_ask({**body, "generate": False})


def api_rerank_all(body: dict[str, Any]) -> dict[str, Any]:
    """Exercise 3.5 over all golden cases: same chunk set, new order."""
    top_k = _int(body.get("top_k"), 5)
    rows = []
    for case_id, pair in GOLDEN.items():
        expected = pair["expected_answer"]
        before = [record["text"] for record in retrieve(pair["question"], top_k, False)]
        after = [record["text"] for record in retrieve(pair["question"], top_k, True)]
        rows.append(
            {
                "id": case_id,
                "difficulty": pair["difficulty"],
                "recall_before": EVALUATOR.evaluate_context_recall(before, expected),
                "recall_after": EVALUATOR.evaluate_context_recall(after, expected),
                "precision_before": EVALUATOR.evaluate_context_precision(before, expected),
                "precision_after": EVALUATOR.evaluate_context_precision(after, expected),
            }
        )
    return {"top_k": top_k, "rows": rows}


def api_gate(body: dict[str, Any]) -> dict[str, Any]:
    baseline_id = str(body.get("baseline", ""))
    candidate_id = str(body.get("candidate", ""))
    baseline_results = _eval_results(_run_dir(baseline_id))
    candidate_results = _eval_results(_run_dir(candidate_id))
    regression = RUNNER.run_regression(candidate_results, baseline_results)
    base_report = RUNNER.generate_report(baseline_results)
    cand_report = RUNNER.generate_report(candidate_results)

    metrics = []
    for metric in ANSWER_METRICS:
        base = regression[f"baseline_avg_{metric}"]
        cand = regression[f"new_avg_{metric}"]
        metrics.append(
            {
                "metric": metric,
                "baseline": base,
                "candidate": cand,
                "delta": cand - base,
                "policy": "block",
                "status": "regressed" if metric in regression["regressions"] else "ok",
            }
        )
    for metric, key in (
        ("context_recall", "avg_context_recall"),
        ("context_precision", "avg_context_precision"),
        ("pass_rate", "pass_rate"),
    ):
        base, cand = base_report[key], cand_report[key]
        delta = None if base is None or cand is None else cand - base
        metrics.append(
            {
                "metric": metric,
                "baseline": base,
                "candidate": cand,
                "delta": delta,
                "policy": "alert",
                "status": "alert" if delta is not None and delta < -REGRESSION_THRESHOLD else "ok",
            }
        )

    cases = []
    baseline_by_id = {r.qa_pair.metadata["id"]: r for r in baseline_results}
    for result in candidate_results:
        case_id = result.qa_pair.metadata["id"]
        base = baseline_by_id.get(case_id)
        difficulty = result.qa_pair.metadata.get("difficulty")
        deltas = {
            metric: getattr(result, metric) - getattr(base, metric) if base else None
            for metric in ANSWER_METRICS
        }
        overall_delta = result.overall_score() - base.overall_score() if base else None
        flags = []
        if base and base.passed and not result.passed:
            flags.append("pass→fail")
        if base and not base.passed and result.passed:
            flags.append("fail→pass")
        for metric in ("faithfulness", "completeness"):
            if deltas[metric] is not None and deltas[metric] < -0.2:
                flags.append(f"{metric} −{abs(deltas[metric]):.2f}")
        needs_review = difficulty in REVIEW_DIFFICULTIES and (
            not result.passed or (overall_delta is not None and overall_delta < -REGRESSION_THRESHOLD)
        )
        cases.append(
            {
                "id": case_id,
                "difficulty": difficulty,
                "attack_type": result.qa_pair.metadata.get("attack_type"),
                "baseline_overall": base.overall_score() if base else None,
                "candidate_overall": result.overall_score(),
                "delta": overall_delta,
                "metric_deltas": deltas,
                "baseline_passed": base.passed if base else None,
                "candidate_passed": result.passed,
                "failure_type": result.failure_type,
                "flags": flags,
                "needs_review": needs_review,
            }
        )

    hard_regressions = [
        case["id"]
        for case in cases
        if case["difficulty"] in REVIEW_DIFFICULTIES and "pass→fail" in case["flags"]
    ]
    if regression["regressions"] or hard_regressions:
        verdict = "BLOCK"
    elif any(case["needs_review"] for case in cases) or any(
        metric["status"] == "alert" for metric in metrics
    ):
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    return {
        "baseline": baseline_id,
        "candidate": candidate_id,
        "threshold": REGRESSION_THRESHOLD,
        "verdict": verdict,
        "regressions": regression["regressions"],
        "hard_regressions": hard_regressions,
        "metrics": metrics,
        "cases": cases,
    }


def api_create_run(body: dict[str, Any]) -> dict[str, Any]:
    return create_live_run(
        str(body.get("prompt_version", "1.0")),
        _int(body.get("top_k"), 5),
        str(body.get("label", "")).strip()[:80],
    )


GET_ROUTES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "/api/status": api_status,
    "/api/cases": api_cases,
    "/api/runs": api_runs,
    "/api/run": api_run,
    "/api/case": api_case,
}
POST_ROUTES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "/api/ask": api_ask,
    "/api/retrieve": api_retrieve,
    "/api/rerank-all": api_rerank_all,
    "/api/gate": api_gate,
    "/api/runs": api_create_run,
}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            params = {key: values[0] for key, values in parse_qs(url.query).items()}
            self._dispatch(GET_ROUTES, url.path, params)
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            self._send_json({"error": "Body phải là JSON object"}, HTTPStatus.BAD_REQUEST)
            return
        self._dispatch(POST_ROUTES, url.path, body)

    def _dispatch(
        self,
        routes: dict[str, Callable[[dict[str, Any]], Any]],
        path: str,
        payload: dict[str, Any],
    ) -> None:
        handler = routes.get(path)
        if handler is None:
            self._send_json({"error": f"Không có endpoint {path}"}, HTTPStatus.NOT_FOUND)
            return
        try:
            self._send_json(handler(payload))
        except ApiError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except Exception as exc:  # surface OpenAI / artifact errors to the UI
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description="OrbitTech evaluation demo UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8014)
    args = parser.parse_args()

    seed_runs()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"OrbitTech Eval Demo: http://{args.host}:{args.port}  (Ctrl+C để dừng)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
