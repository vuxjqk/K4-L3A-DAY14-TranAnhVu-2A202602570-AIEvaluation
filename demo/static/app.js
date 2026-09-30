"use strict";

// ------------------------------------------------------------------ state
const state = {
  status: null,
  cases: [],
  runs: [],
  runId: null,
  runResults: [],
  source: "golden",
  mode: "saved",
  filters: { difficulty: "all", failure: "all", status: "all" },
  selectedCase: null,
  compareMode: "rerank",
};

const NOTES_RUN = "run-001"; // case_notes.json được viết từ run baseline này
const DIFFICULTIES = ["easy", "medium", "hard", "adversarial"];
const FAILURE_TYPES = ["hallucination", "irrelevant", "incomplete", "off_topic"];
const VERDICT_LABEL = {
  wrong_answer: ["Sai thật — lỗi generation", "bad"],
  retrieval_miss: ["Retrieval thiếu / sai evidence", "warn"],
  metric_false_negative: ["False negative của metric", "accent"],
};

// ---------------------------------------------------------------- helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

function fmtDelta(value) {
  if (value === null || value === undefined) return "—";
  const cls = value < -0.0005 ? "delta-neg" : value > 0.0005 ? "delta-pos" : "";
  return `<span class="${cls}">${value > 0 ? "+" : ""}${value.toFixed(3)}</span>`;
}

function band(value) {
  if (value === null || value === undefined) return "";
  return value >= 0.8 ? "good" : value >= 0.6 ? "warn" : "bad";
}

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 4500);
}

function loading(container, text = "Đang chạy…") {
  container.innerHTML = `<div class="loading"><span class="spinner"></span>${esc(text)}</div>`;
}

async function withBusy(button, fn) {
  button.disabled = true;
  try { await fn(); } catch (err) { toast(err.message); } finally { button.disabled = false; }
}

function caseById(id) { return state.cases.find((c) => c.id === id); }

function diffPill(difficulty) {
  const cls = { easy: "good", medium: "accent", hard: "warn", adversarial: "bad" }[difficulty] || "";
  return difficulty ? `<span class="pill ${cls}">${esc(difficulty)}</span>` : "";
}

function passPill(passed, failureType) {
  if (passed === null || passed === undefined) return "";
  return passed
    ? `<span class="pill good">PASS</span>`
    : `<span class="pill bad">FAIL${failureType ? ` · ${esc(failureType)}` : ""}</span>`;
}

function caseOption(c) {
  const text = c.question.length > 70 ? `${c.question.slice(0, 67)}…` : c.question;
  return `<option value="${c.id}">${c.id} · ${esc(text)}</option>`;
}

function runOptions(selected) {
  return state.runs.map((r) => (
    `<option value="${r.id}" ${r.id === selected ? "selected" : ""}>${r.id} · ${esc(r.label)} (prompt ${esc(r.prompt_version)}, pass ${(r.summary.pass_rate * 100).toFixed(0)}%)</option>`
  )).join("");
}

// ----------------------------------------------------------- highlighting
// Marks BM25-matched words and underlines gold-evidence spans inside a chunk.
function highlight(text, words, evidences) {
  const spans = [];
  for (const evidence of evidences || []) {
    const start = text.indexOf(evidence);
    if (start >= 0) spans.push([start, start + evidence.length]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  const pattern = words && words.length
    ? new RegExp(`\\b(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi")
    : null;
  const mark = (segment) => {
    const escaped = esc(segment);
    return pattern ? escaped.replace(pattern, "<mark>$1</mark>") : escaped;
  };
  let html = "";
  let cursor = 0;
  for (const [start, end] of spans) {
    if (start < cursor) continue;
    html += mark(text.slice(cursor, start));
    html += `<span class="gold-span" title="Gold evidence">${mark(text.slice(start, end))}</span>`;
    cursor = end;
  }
  return html + mark(text.slice(cursor));
}

// -------------------------------------------------------------- renderers
function metricCard(name, value, sub) {
  const b = band(value);
  const width = value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return `<div class="metric">
    <div class="name">${esc(name)}</div>
    <div class="value band-${b}">${fmt(value)}</div>
    <div class="bar"><span class="fill-${b}" style="width:${width}%"></span><i class="threshold" title="ngưỡng pass 0.5"></i></div>
    ${sub ? `<div class="sub">${esc(sub)}</div>` : ""}
  </div>`;
}

function renderChunks(chunks, { hasReference }) {
  if (!chunks.length) return `<div class="empty">Retriever không trả về chunk nào có điểm &gt; 0.</div>`;
  return `<div class="chunk-list">${chunks.map((c) => {
    const moved = c.bm25_rank !== c.rank;
    const move = moved
      ? `<span class="${c.bm25_rank > c.rank ? "move-up" : "move-down"}">${c.bm25_rank > c.rank ? "▲" : "▼"} từ #${c.bm25_rank}</span>`
      : "";
    const relevance = hasReference && c.relevant !== undefined
      ? `<span class="pill ${c.relevant ? "good" : ""}" title="Tỷ lệ token của reference có trong chunk (ngưỡng 0.1)">${c.relevant ? "relevant" : "noise"} · ${fmt(c.coverage, 2)}</span>`
      : "";
    const gold = c.gold_evidence && c.gold_evidence.length
      ? `<span class="pill good">chứa gold evidence</span>` : "";
    const terms = c.matched_terms.length
      ? `<span class="muted">khớp BM25:</span><span class="terms">${c.matched_terms.map((t) => `<span class="term">${esc(t)}</span>`).join("")}</span>`
      : "";
    const cls = hasReference && c.relevant !== undefined ? (c.relevant ? "relevant" : "noise") : "";
    return `<div class="chunk ${cls}">
      <div class="chunk-head">
        <span class="rank">#${c.rank}</span>${move}
        <span class="chunk-id">${esc(c.chunk_id)}</span>
        <span class="pill">${esc(c.source_doc)}</span>
        <span class="pill" title="BM25 score (đã áp dụng source diversity decay)">score ${fmt(c.score, 2)}</span>
        ${relevance}${gold}${terms}
      </div>
      <div class="chunk-text">${highlight(c.text, c.highlight, c.gold_evidence)}</div>
    </div>`;
  }).join("")}</div>`;
}

function renderSource(source) {
  if (!source) return "";
  const parts = source.kind === "saved"
    ? [`answer đã lưu · ${source.run_id}`, `model ${source.model}`, `top_k ${source.top_k}`, `prompt ${source.prompt_version}`]
    : ["live", `model ${source.model || "?"}`, `top_k ${source.top_k}`, `prompt ${source.prompt_version}`, source.rerank ? "rerank bật" : "rerank tắt"];
  return parts.map((p) => `<span class="pill">${esc(p)}</span>`).join(" ");
}

function renderQueryTerms(view) {
  const changed = (view.query_terms || []).filter((t) => t.raw !== t.norm);
  const all = (view.query_terms || []).map((t) => (
    t.raw === t.norm ? `<span class="term">${esc(t.raw)}</span>` : `<span class="term" title="retriever normalize">${esc(t.raw)} → ${esc(t.norm)}</span>`
  )).join("");
  return `<div class="small"><span class="muted">Query terms sau khi bỏ stopword${changed.length ? " (có normalize)" : ""}:</span> <span class="terms">${all}</span></div>`;
}

function renderExplanation(view) {
  const ev = view.evaluation;
  const note = view.note;
  const blocks = [];

  if (note) {
    const [label, cls] = VERDICT_LABEL[note.verdict] || [note.verdict, ""];
    const stale = view.source && (view.source.kind !== "saved" || view.source.run_id !== NOTES_RUN);
    blocks.push(`<div>
      <h5>Phân tích (reflection.md)</h5>
      <p style="margin:0 0 6px"><span class="pill ${cls}">${esc(label)}</span> <b>${esc(note.title)}</b></p>
      <p style="margin:0">${esc(note.summary)}</p>
      <ul>${note.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
      <p style="margin:6px 0 0"><b>Fix đề xuất:</b> ${esc(note.fix)}</p>
      ${stale ? `<p class="small muted" style="margin:6px 0 0">Ghi chú viết cho answer của ${NOTES_RUN}; answer đang xem có thể khác — đối chiếu với phần kiểm tra tự động bên dưới.</p>` : ""}
    </div>`);
  }

  if (ev && ev.root_cause) {
    blocks.push(`<div><h5>find_root_cause()</h5><code>${esc(ev.root_cause)}</code></div>`);
  }

  if (view.gold_evidence) {
    blocks.push(`<div><h5>Gold evidence có được retrieve không?</h5><div class="evidence-list">${
      view.gold_evidence.map((g) => `<div class="evidence">
        ${g.rank ? `<span class="pill good">có · chunk #${g.rank}</span>` : `<span class="pill bad">không được retrieve</span>`}
        <span class="pill">${esc(g.source_doc)}</span>
        <div style="margin-top:4px">${esc(g.text)}</div>
      </div>`).join("")
    }</div></div>`);
  }

  if (view.numbers) {
    const { missing, unsupported } = view.numbers;
    const items = [];
    for (const m of missing) {
      items.push(`<li><b>${esc(m.value)}</b> có trong expected nhưng thiếu trong answer${m.ranks.length ? ` — dù có ở chunk ${m.ranks.map((r) => `#${r}`).join(", ")}` : " — và không có trong chunk nào"}</li>`);
    }
    for (const u of unsupported) {
      items.push(`<li><b>${esc(u)}</b> xuất hiện trong answer nhưng không có trong expected hay bất kỳ chunk nào (không có căn cứ)</li>`);
    }
    blocks.push(`<div><h5>Kiểm tra số liệu (ngày, USD, %)</h5>${
      items.length ? `<ul>${items.join("")}</ul>` : `<p class="small muted" style="margin:0">Không phát hiện số liệu thiếu hoặc không có căn cứ.</p>`
    }<p class="small muted" style="margin:4px 0 0">Word-overlap không phân biệt "21 days" với "45 days"; phần này bổ sung góc nhìn đó.</p></div>`);
  }

  const lexical = (view.chunks || []).filter((c) => c.matched_terms.length === 1);
  if (lexical.length && lexical.length === view.chunks.length) {
    blocks.push(`<div class="notice warn">Mọi chunk chỉ khớp đúng một term (${[...new Set(lexical.map((c) => c.matched_terms[0]))].map(esc).join(", ")}) — dấu hiệu retrieval khớp lexical ngẫu nhiên, không phải khớp ý nghĩa.</div>`);
  }

  if (!blocks.length) return "";
  return `<div class="explain" id="explain-box">
    <header><strong>Giải thích failure</strong>${ev ? passPill(ev.passed, ev.failure_type) : ""}</header>
    <div class="body">${blocks.join("")}</div>
  </div>`;
}

function renderCaseView(view, { autoExplain = false } = {}) {
  const ev = view.evaluation;
  const c = view.case_id ? caseById(view.case_id) : null;
  const hasReference = Boolean(view.expected_answer);

  let evaluation = "";
  if (ev) {
    const basis = view.eval_basis === "golden"
      ? `Chấm theo golden ${esc(view.case_id)}: Faithfulness so với gold evidence, Completeness so với expected answer.`
      : `Chấm theo <b>reference bạn nhập</b>: không có gold evidence nên Faithfulness được đo với các chunk đã retrieve.`;
    const failureButton = ev.passed
      ? `<span class="pill good">PASS</span>`
      : `<button class="pill bad" id="failure-toggle" title="Mở phần giải thích">FAIL · ${esc(ev.failure_type)} ▾</button>`;
    const adversarial = view.difficulty === "adversarial"
      ? `<div class="notice warn" style="margin-top:12px">Case adversarial: word-overlap thường phạt câu từ chối đúng. Đọc answer trước khi kết luận từ điểm số.</div>`
      : "";
    evaluation = `<div class="card">
      <div class="card-head"><h2>Đánh giá</h2><span class="spacer"></span>${failureButton}</div>
      <p class="small muted" style="margin:0 0 10px">${basis}</p>
      <div class="side-by-side">
        <div><h4>${view.eval_basis === "golden" ? "Expected answer" : "Reference của bạn"}</h4><div class="answer-text">${esc(view.expected_answer)}</div></div>
        <div><h4>Actual answer</h4><div class="answer-text">${esc(view.answer)}</div></div>
      </div>
      <div class="metrics">
        ${metricCard("Faithfulness", ev.faithfulness, view.eval_basis === "golden" ? "vs gold evidence" : "vs retrieved chunks")}
        ${metricCard("Relevance", ev.relevance, "vs question")}
        ${metricCard("Completeness", ev.completeness, view.eval_basis === "golden" ? "vs expected" : "vs reference")}
        ${metricCard("Overall", ev.overall, "mean F/R/C")}
        ${metricCard("Context Recall", ev.context_recall, "retrieval · không tính vào overall")}
        ${metricCard("Context Precision", ev.context_precision, "AP@K · không tính vào overall")}
      </div>
      ${adversarial}
      <div id="explain-slot" ${autoExplain || ev.passed ? "" : "hidden"}>${renderExplanation(view)}</div>
    </div>`;
  } else if (view.answer) {
    evaluation = `<div class="notice">Câu hỏi tự nhập chưa có expected answer hay gold context nên <b>không chấm</b> Faithfulness / Completeness — con số lúc này sẽ không có căn cứ. Thêm reference answer để chấm, hoặc chọn một case golden.</div>`;
  }

  const answerCard = view.answer
    ? `<div class="card">
        <div class="card-head"><h2>Câu trả lời</h2>${c ? diffPill(c.difficulty) : `<span class="pill">câu hỏi tự nhập</span>`}<span class="spacer"></span>${renderSource(view.source)}</div>
        <p class="muted small" style="margin:0 0 8px">${esc(view.question)}</p>
        <div class="answer-text">${esc(view.answer)}</div>
      </div>`
    : `<div class="card"><div class="card-head"><h2>Câu trả lời</h2><span class="spacer"></span>${renderSource(view.source)}</div>
        <p class="muted small" style="margin:0 0 8px">${esc(view.question)}</p>
        <div class="notice warn">${esc(view.warning || "Chưa sinh câu trả lời.")}</div></div>`;

  const retrieval = view.retrieval && !ev
    ? `<span class="pill">Recall ${fmt(view.retrieval.context_recall)}</span><span class="pill">Precision ${fmt(view.retrieval.context_precision)}</span>`
    : "";
  const chunksCard = `<div class="card">
    <div class="card-head"><h2>Retrieved chunks</h2><span class="pill">${view.chunks.length} chunks</span>${retrieval}<span class="spacer"></span>
      <span class="small muted"><mark>highlight</mark> = từ khớp BM25 · <span class="gold-span">gạch chân</span> = gold evidence</span></div>
    ${renderQueryTerms(view)}
    <div style="margin-top:10px">${renderChunks(view.chunks, { hasReference })}</div>
  </div>`;

  return answerCard + evaluation + chunksCard;
}

function bindExplainToggle(container) {
  const button = $("#failure-toggle", container);
  const slot = $("#explain-slot", container);
  if (button && slot) {
    button.addEventListener("click", () => {
      slot.hidden = !slot.hidden;
      if (!slot.hidden) slot.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }
}

// ------------------------------------------------------------- playground
function setSegmented(groupId, attr, value) {
  $$(`#${groupId} .seg`).forEach((b) => b.classList.toggle("active", b.dataset[attr] === value));
}

function updatePlaygroundControls() {
  $("#golden-input").hidden = state.source !== "golden";
  $("#custom-input").hidden = state.source !== "custom";
  if (state.source === "custom" && state.mode === "saved") state.mode = "live";
  setSegmented("source-toggle", "source", state.source);
  setSegmented("mode-toggle", "mode", state.mode);
  $("#mode-toggle [data-mode='saved']").disabled = state.source === "custom";
  $("#live-options").disabled = state.mode !== "live";

  const hint = state.mode === "saved"
    ? `Lấy answer và chunks từ ${state.runId || "run đang xem"} — không gọi OpenAI.`
    : state.status && state.status.has_key
      ? "Retrieve bằng BM25 rồi gọi OpenAI để sinh answer mới."
      : "Không có OPENAI_API_KEY: chỉ chạy retrieval, không sinh answer.";
  $("#mode-hint").textContent = hint;

  const c = caseById($("#case-select").value);
  $("#case-question").textContent = c ? c.question : "";
}

async function runPlayground() {
  const out = $("#playground-result");
  const button = $("#run-btn");
  await withBusy(button, async () => {
    loading(out, state.mode === "live" ? "Đang retrieve và gọi model…" : "Đang tải trace…");
    let view;
    if (state.source === "golden" && state.mode === "saved") {
      view = await api(`/api/case?run=${state.runId}&case=${$("#case-select").value}`);
    } else {
      const body = {
        top_k: Number($("#opt-topk").value),
        rerank: $("#opt-rerank").checked,
        prompt_version: $("#opt-prompt").value,
      };
      if (state.source === "golden") body.case_id = $("#case-select").value;
      else {
        body.question = $("#custom-question").value;
        body.reference = $("#custom-reference").value;
      }
      view = await api("/api/ask", body);
    }
    out.innerHTML = renderCaseView(view);
    bindExplainToggle(out);
  }).catch(() => {});
  if (!out.innerHTML.includes("card")) out.innerHTML = `<div class="empty">Không chạy được — xem thông báo lỗi.</div>`;
}

// --------------------------------------------------------------- explorer
function renderFilterChips() {
  const make = (id, key, values, labels = {}) => {
    $(`#${id}`).innerHTML = ["all", ...values].map((v) => (
      `<button class="chip ${state.filters[key] === v ? "active" : ""}" data-key="${key}" data-value="${v}">${esc(labels[v] || (v === "all" ? "Tất cả" : v))}</button>`
    )).join("");
  };
  make("filter-difficulty", "difficulty", DIFFICULTIES);
  make("filter-failure", "failure", FAILURE_TYPES);
  make("filter-status", "status", ["failed", "passed"], { failed: "Fail", passed: "Pass" });
}

function filteredResults() {
  const f = state.filters;
  return state.runResults.filter((r) => (
    (f.difficulty === "all" || r.difficulty === f.difficulty)
    && (f.failure === "all" || r.failure_type === f.failure)
    && (f.status === "all" || (f.status === "passed") === r.passed)
  ));
}

function renderExplorer() {
  renderFilterChips();
  const rows = filteredResults();
  const failed = rows.filter((r) => !r.passed).length;
  $("#explorer-summary").innerHTML = `
    <span class="pill">${rows.length} / ${state.runResults.length} case</span>
    <span class="pill bad">${failed} fail</span>
    <span class="pill good">${rows.length - failed} pass</span>
    <span class="pill accent">run ${esc(state.runId)}</span>`;
  $("#explorer-table").innerHTML = `
    <thead><tr><th>ID</th><th>Diff.</th><th>Question</th>
      <th class="num">Faith.</th><th class="num">Relev.</th><th class="num">Compl.</th><th class="num">Overall</th>
      <th class="num">Ctx R</th><th class="num">Ctx P</th><th>Kết quả</th><th></th></tr></thead>
    <tbody>${rows.map((r) => {
      const c = caseById(r.id);
      return `<tr class="clickable ${state.selectedCase === r.id ? "selected" : ""}" data-id="${r.id}">
        <td><b>${r.id}</b></td><td>${diffPill(r.difficulty)}</td>
        <td class="q-cell">${esc(r.question)}</td>
        ${["faithfulness", "relevance", "completeness", "overall", "context_recall", "context_precision"].map((k) => (
          `<td class="num band-${band(r[k])}">${fmt(r[k])}</td>`)).join("")}
        <td>${passPill(r.passed, r.failure_type)}</td>
        <td>${c && c.has_note ? `<span class="pill accent" title="Có phân tích trong reflection">phân tích</span>` : ""}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="11" class="muted">Không có case nào khớp bộ lọc.</td></tr>`}</tbody>`;
}

async function openExplorerCase(id) {
  state.selectedCase = id;
  renderExplorer();
  const detail = $("#explorer-detail");
  loading(detail, `Đang mở ${id}…`);
  try {
    const view = await api(`/api/case?run=${state.runId}&case=${id}`);
    detail.innerHTML = `<div class="results">${renderCaseView(view, { autoExplain: true })}</div>`;
    bindExplainToggle(detail);
    detail.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    detail.innerHTML = "";
    toast(err.message);
  }
}

async function loadRun(runId) {
  state.runId = runId;
  const data = await api(`/api/run?id=${runId}`);
  state.runResults = data.results;
  state.selectedCase = null;
  $("#explorer-detail").innerHTML = "";
  renderExplorer();
  updatePlaygroundControls();
}

// ---------------------------------------------------------------- compare
function miniChunks(chunks, showMove) {
  return chunks.map((c) => {
    const move = showMove && c.bm25_rank !== c.rank
      ? `<span class="${c.bm25_rank > c.rank ? "move-up" : "move-down"}">${c.bm25_rank > c.rank ? "▲" : "▼"}${Math.abs(c.bm25_rank - c.rank)}</span>` : "";
    return `<div class="mini-chunk ${c.relevant ? "relevant" : ""}">
      <b>#${c.rank}</b>
      <span><span class="chunk-id">${esc(c.chunk_id)}</span> <span class="muted small">${esc(c.text.slice(0, 90))}…</span></span>
      <span>${move} <span class="pill ${c.relevant ? "good" : ""}">${c.relevant ? "relevant" : "noise"}</span></span>
    </div>`;
  }).join("");
}

async function runCompareRerank() {
  const out = $("#cmp-rerank-result");
  loading(out);
  const body = { case_id: $("#cmp-case").value, top_k: Number($("#cmp-topk").value) };
  try {
    const [before, after] = await Promise.all([
      api("/api/retrieve", { ...body, rerank: false }),
      api("/api/retrieve", { ...body, rerank: true }),
    ]);
    const dp = after.retrieval.context_precision - before.retrieval.context_precision;
    const dr = after.retrieval.context_recall - before.retrieval.context_recall;
    const verdict = Math.abs(dp) < 0.0005
      ? `<div class="notice">Thứ tự chunk liên quan không đổi nên Context Precision giữ nguyên.</div>`
      : dp > 0
        ? `<div class="notice good">Reranking đưa chunk liên quan lên sớm hơn → Precision tăng ${dp.toFixed(3)}.</div>`
        : `<div class="notice bad">Reranking đẩy chunk liên quan xuống → Precision giảm ${Math.abs(dp).toFixed(3)}. Lexical overlap với câu hỏi không đồng nghĩa với liên quan tới đáp án.</div>`;
    out.innerHTML = `
      <p class="question-preview">${esc(before.question)}</p>
      <div class="compare-cols">
        <div><div class="col-head"><b>BM25 (không rerank)</b>
          <span><span class="pill">Recall ${fmt(before.retrieval.context_recall)}</span> <span class="pill">Precision ${fmt(before.retrieval.context_precision)}</span></span></div>
          ${miniChunks(before.chunks, false)}</div>
        <div><div class="col-head"><b>Sau rerank_by_overlap()</b>
          <span><span class="pill">Recall ${fmt(after.retrieval.context_recall)} (${fmtDelta(dr)})</span> <span class="pill">Precision ${fmt(after.retrieval.context_precision)} (${fmtDelta(dp)})</span></span></div>
          ${miniChunks(after.chunks, true)}</div>
      </div>
      <div style="margin-top:12px">${verdict}</div>`;
  } catch (err) {
    out.innerHTML = "";
    toast(err.message);
  }
}

async function runCompareAll(button) {
  const out = $("#cmp-all-result");
  await withBusy(button, async () => {
    loading(out, "Đang rerank 20 case…");
    const data = await api("/api/rerank-all", { top_k: Number($("#cmp-topk").value) });
    const avg = (key) => data.rows.reduce((s, r) => s + r[key], 0) / data.rows.length;
    const up = data.rows.filter((r) => r.precision_after - r.precision_before > 0.0005).length;
    const down = data.rows.filter((r) => r.precision_after - r.precision_before < -0.0005).length;
    out.innerHTML = `<h3>Toàn bộ golden dataset (top_k ${data.top_k})</h3>
      <div class="summary-row"><span class="pill good">${up} tăng</span><span class="pill bad">${down} giảm</span>
        <span class="pill">${data.rows.length - up - down} không đổi</span>
        <span class="pill">Precision TB ${fmt(avg("precision_before"))} → ${fmt(avg("precision_after"))}</span>
        <span class="pill">Recall TB ${fmt(avg("recall_before"))} → ${fmt(avg("recall_after"))}</span></div>
      <div class="table-wrap"><table class="data"><thead><tr><th>ID</th><th>Diff.</th>
        <th class="num">Recall trước</th><th class="num">Recall sau</th><th class="num">Precision trước</th><th class="num">Precision sau</th><th class="num">Δ Precision</th></tr></thead>
      <tbody>${data.rows.map((r) => `<tr class="clickable" data-cmp-case="${r.id}"><td><b>${r.id}</b></td><td>${diffPill(r.difficulty)}</td>
        <td class="num">${fmt(r.recall_before)}</td><td class="num">${fmt(r.recall_after)}</td>
        <td class="num">${fmt(r.precision_before)}</td><td class="num">${fmt(r.precision_after)}</td>
        <td class="num">${fmtDelta(r.precision_after - r.precision_before)}</td></tr>`).join("")}</tbody></table></div>`;
  });
}

async function runCompareRuns() {
  const out = $("#cmp-runs-result");
  const a = $("#cmp-run-a").value;
  const b = $("#cmp-run-b").value;
  loading(out);
  try {
    const [runA, runB] = await Promise.all([api(`/api/run?id=${a}`), api(`/api/run?id=${b}`)]);
    const byId = Object.fromEntries(runB.results.map((r) => [r.id, r]));
    const summaryRow = (key, label) => {
      const va = runA.summary[key];
      const vb = runB.summary[key];
      return `<tr><td>${label}</td><td class="num">${fmt(va)}</td><td class="num">${fmt(vb)}</td><td class="num">${va == null || vb == null ? "—" : fmtDelta(vb - va)}</td></tr>`;
    };
    out.innerHTML = `
      ${a === b ? `<div class="notice warn" style="margin-top:12px">Đang so một run với chính nó. Tạo thêm run ở tab Quality Gate để so sánh.</div>` : ""}
      <div class="grid-2" style="margin-top:12px">
        <div><h3>Tổng hợp</h3><table class="data"><thead><tr><th>Metric</th><th class="num">${a}</th><th class="num">${b}</th><th class="num">Δ</th></tr></thead><tbody>
          ${summaryRow("pass_rate", "Pass rate")}${summaryRow("avg_faithfulness", "Faithfulness")}${summaryRow("avg_relevance", "Relevance")}
          ${summaryRow("avg_completeness", "Completeness")}${summaryRow("avg_context_recall", "Context Recall")}${summaryRow("avg_context_precision", "Context Precision")}
        </tbody></table></div>
        <div><h3>Cấu hình</h3><table class="data"><tbody>
          ${["label", "prompt_version", "top_k", "model", "generated_at"].map((k) => `<tr><td>${k}</td><td>${esc(runA.meta[k])}</td><td>${esc(runB.meta[k])}</td></tr>`).join("")}
        </tbody></table></div>
      </div>
      <h3>Theo case (bấm để xem answer và thứ tự chunk)</h3>
      <div class="table-wrap"><table class="data"><thead><tr><th>ID</th><th>Diff.</th><th class="num">Overall A</th><th class="num">Overall B</th><th class="num">Δ</th><th>A</th><th>B</th></tr></thead>
      <tbody>${runA.results.map((ra) => {
        const rb = byId[ra.id];
        return `<tr class="clickable" data-run-case="${ra.id}"><td><b>${ra.id}</b></td><td>${diffPill(ra.difficulty)}</td>
          <td class="num">${fmt(ra.overall)}</td><td class="num">${fmt(rb && rb.overall)}</td><td class="num">${rb ? fmtDelta(rb.overall - ra.overall) : "—"}</td>
          <td>${passPill(ra.passed, ra.failure_type)}</td><td>${rb ? passPill(rb.passed, rb.failure_type) : ""}</td></tr>`;
      }).join("")}</tbody></table></div>
      <div id="cmp-run-case"></div>`;
  } catch (err) {
    out.innerHTML = "";
    toast(err.message);
  }
}

async function openRunCase(id) {
  const out = $("#cmp-run-case");
  const a = $("#cmp-run-a").value;
  const b = $("#cmp-run-b").value;
  loading(out);
  try {
    const [va, vb] = await Promise.all([api(`/api/case?run=${a}&case=${id}`), api(`/api/case?run=${b}&case=${id}`)]);
    const orderA = va.chunks.map((c) => c.chunk_id);
    const col = (label, view, other) => `<div class="card">
      <div class="col-head"><b>${label}</b>${passPill(view.evaluation.passed, view.evaluation.failure_type)}
        <span class="pill">overall ${fmt(view.evaluation.overall)}</span><span class="pill">Precision ${fmt(view.evaluation.context_precision)}</span></div>
      <div class="answer-text" style="margin-bottom:10px">${esc(view.answer)}</div>
      ${view.chunks.map((c) => {
        const changed = other && other[c.rank - 1] !== c.chunk_id;
        return `<div class="mini-chunk ${c.relevant ? "relevant" : ""}"><b>#${c.rank}</b>
          <span><span class="chunk-id">${esc(c.chunk_id)}</span> <span class="muted small">${esc(c.source_doc)}</span></span>
          <span>${changed ? `<span class="pill warn">thứ tự khác</span>` : ""}</span></div>`;
      }).join("")}</div>`;
    out.innerHTML = `<h3>${id} — ${esc(va.question)}</h3><div class="compare-cols">${col(a, va, null)}${col(b, vb, orderA)}</div>`;
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    out.innerHTML = "";
    toast(err.message);
  }
}

// ------------------------------------------------------------------- gate
async function runGate(button) {
  const out = $("#gate-result");
  await withBusy(button, async () => {
    loading(out, "Đang đánh giá lại hai run bằng evaluation core…");
    const g = await api("/api/gate", { baseline: $("#gate-baseline").value, candidate: $("#gate-candidate").value });
    const reasons = {
      BLOCK: `Block deploy: ${[
        g.regressions.length ? `metric giảm > ${g.threshold}: ${g.regressions.join(", ")}` : "",
        g.hard_regressions.length ? `case Hard/Adversarial pass→fail: ${g.hard_regressions.join(", ")}` : "",
      ].filter(Boolean).join(" · ")}`,
      REVIEW: "Không có regression bị chặn, nhưng còn case Hard/Adversarial fail hoặc metric cảnh báo cần người review.",
      PASS: "Không có regression và không có case Hard/Adversarial cần review.",
    };
    const review = g.cases.filter((c) => c.needs_review);
    out.innerHTML = `
      <div class="verdict ${g.verdict}"><b>${g.verdict}</b><span>${esc(reasons[g.verdict])}</span></div>
      ${g.baseline === g.candidate ? `<div class="notice warn">Baseline và candidate là cùng một run — delta luôn bằng 0. Tạo run mới để kiểm tra thay đổi.</div>` : ""}
      <div class="card"><h2>Metric so với baseline (${esc(g.baseline)} → ${esc(g.candidate)})</h2>
        <table class="data"><thead><tr><th>Metric</th><th>Chính sách</th><th class="num">Baseline</th><th class="num">Candidate</th><th class="num">Δ</th><th>Trạng thái</th></tr></thead>
        <tbody>${g.metrics.map((m) => `<tr class="${m.status !== "ok" ? "review" : ""}"><td><b>${m.metric}</b></td>
          <td><span class="pill ${m.policy === "block" ? "bad" : "warn"}">${m.policy}</span></td>
          <td class="num">${fmt(m.baseline)}</td><td class="num">${fmt(m.candidate)}</td><td class="num">${fmtDelta(m.delta)}</td>
          <td>${m.status === "ok" ? `<span class="pill good">ok</span>` : `<span class="pill ${m.status === "regressed" ? "bad" : "warn"}">${m.status}</span>`}</td></tr>`).join("")}</tbody></table>
      </div>
      <div class="card"><h2>Case Hard / Adversarial cần review (${review.length})</h2>
        ${review.length ? `<div class="summary-row">${review.map((c) => `<button class="pill ${c.difficulty === "adversarial" ? "bad" : "warn"}" data-open-case="${c.id}">${c.id} · ${fmt(c.candidate_overall)}${c.failure_type ? ` · ${esc(c.failure_type)}` : ""}</button>`).join("")}</div>
          <p class="small muted">Case adversarial fail theo word-overlap có thể là refusal đúng (A02, A03) — cần đọc answer. Bấm để mở trong Failure Explorer.</p>` : `<p class="muted">Không có.</p>`}
      </div>
      <div class="card"><h2>Tất cả case</h2><div class="table-wrap"><table class="data">
        <thead><tr><th>ID</th><th>Diff.</th><th class="num">Baseline</th><th class="num">Candidate</th><th class="num">Δ overall</th>
          <th class="num">Δ F</th><th class="num">Δ R</th><th class="num">Δ C</th><th>Kết quả</th><th>Cờ</th></tr></thead>
        <tbody>${g.cases.map((c) => `<tr class="clickable ${c.needs_review ? "review" : ""}" data-open-case="${c.id}"><td><b>${c.id}</b></td><td>${diffPill(c.difficulty)}</td>
          <td class="num">${fmt(c.baseline_overall)}</td><td class="num">${fmt(c.candidate_overall)}</td><td class="num">${fmtDelta(c.delta)}</td>
          <td class="num">${fmtDelta(c.metric_deltas.faithfulness)}</td><td class="num">${fmtDelta(c.metric_deltas.relevance)}</td><td class="num">${fmtDelta(c.metric_deltas.completeness)}</td>
          <td>${passPill(c.candidate_passed, c.failure_type)}</td>
          <td>${c.flags.map((f) => `<span class="pill ${f.startsWith("fail→") ? "good" : "bad"}">${esc(f)}</span>`).join(" ")}${c.needs_review ? ` <span class="pill warn">review</span>` : ""}</td></tr>`).join("")}</tbody>
      </table></div></div>`;
  });
}

async function createRun(button) {
  if (!state.status.has_key) { toast("Cần OPENAI_API_KEY trong .env để tạo run live."); return; }
  const prompt = $("#new-run-prompt").value;
  if (!confirm(`Sinh 20 answers mới bằng OpenAI (prompt ${prompt})?`)) return;
  const out = $("#gate-result");
  await withBusy(button, async () => {
    loading(out, "Đang sinh 20 answers và đánh giá… (~30–60 giây)");
    const meta = await api("/api/runs", {
      prompt_version: prompt,
      top_k: Number($("#new-run-topk").value),
      label: $("#new-run-label").value,
    });
    await refreshRuns();
    $("#gate-candidate").value = meta.id;
    toast(`Đã tạo ${meta.id}`);
    await runGate($("#gate-btn"));
  });
}

async function refreshRuns() {
  state.runs = await api("/api/runs");
  const latest = state.runs.length ? state.runs[state.runs.length - 1].id : null;
  const first = state.runs.length ? state.runs[0].id : null;
  $("#run-select").innerHTML = runOptions(state.runId || first);
  $("#gate-baseline").innerHTML = runOptions(first);
  $("#gate-candidate").innerHTML = runOptions(latest);
  $("#cmp-run-a").innerHTML = runOptions(first);
  $("#cmp-run-b").innerHTML = runOptions(latest);
}

// ------------------------------------------------------------------- init
function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab.dataset.tab}`));
    if (tab.dataset.tab === "compare") {
      if (state.compareMode === "rerank" && !$("#cmp-rerank-result").innerHTML) runCompareRerank();
      if (state.compareMode === "runs") runCompareRuns();
    }
  }));

  $("#run-select").addEventListener("change", (e) => loadRun(e.target.value).catch((err) => toast(err.message)));

  $$("#source-toggle .seg").forEach((b) => b.addEventListener("click", () => {
    state.source = b.dataset.source;
    if (state.source === "golden" && state.mode === "live" && !state.userPickedLive) state.mode = "saved";
    updatePlaygroundControls();
  }));
  $$("#mode-toggle .seg").forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    state.mode = b.dataset.mode;
    state.userPickedLive = state.mode === "live";
    updatePlaygroundControls();
  }));
  $("#case-select").addEventListener("change", updatePlaygroundControls);
  $("#run-btn").addEventListener("click", runPlayground);
  $("#custom-question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runPlayground();
  });

  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) {
      state.filters[chip.dataset.key] = chip.dataset.value;
      renderExplorer();
      return;
    }
    const row = e.target.closest("#explorer-table tr.clickable");
    if (row) { openExplorerCase(row.dataset.id); return; }
    const cmpRow = e.target.closest("[data-cmp-case]");
    if (cmpRow) { $("#cmp-case").value = cmpRow.dataset.cmpCase; runCompareRerank(); window.scrollTo({ top: 0, behavior: "smooth" }); return; }
    const runRow = e.target.closest("[data-run-case]");
    if (runRow) { openRunCase(runRow.dataset.runCase); return; }
    const open = e.target.closest("[data-open-case]");
    if (open) {
      const run = $("#gate-candidate").value;
      $("#run-select").value = run;
      loadRun(run).then(() => {
        $(".tab[data-tab='explorer']").click();
        openExplorerCase(open.dataset.openCase);
      }).catch((err) => toast(err.message));
    }
  });

  $$("#compare-toggle .seg").forEach((b) => b.addEventListener("click", () => {
    state.compareMode = b.dataset.compare;
    setSegmented("compare-toggle", "compare", state.compareMode);
    $("#compare-rerank").hidden = state.compareMode !== "rerank";
    $("#compare-runs").hidden = state.compareMode !== "runs";
    if (state.compareMode === "runs") runCompareRuns();
    else runCompareRerank();
  }));
  $("#cmp-case").addEventListener("change", runCompareRerank);
  $("#cmp-topk").addEventListener("change", () => { runCompareRerank(); $("#cmp-all-result").innerHTML = ""; });
  $("#cmp-all-btn").addEventListener("click", (e) => runCompareAll(e.currentTarget));
  $("#cmp-run-a").addEventListener("change", runCompareRuns);
  $("#cmp-run-b").addEventListener("change", runCompareRuns);

  $("#gate-btn").addEventListener("click", (e) => runGate(e.currentTarget));
  $("#new-run-btn").addEventListener("click", (e) => createRun(e.currentTarget));
}

async function init() {
  bindEvents();
  try {
    const [status, cases] = await Promise.all([api("/api/status"), api("/api/cases")]);
    state.status = status;
    state.cases = cases;
    const keyPill = $("#key-status");
    keyPill.textContent = status.has_key ? `OpenAI ✓ ${status.model || ""}` : "Không có API key";
    keyPill.className = `pill ${status.has_key ? "good" : "warn"}`;
    $("#new-run-btn").disabled = !status.has_key;

    const options = cases.map(caseOption).join("");
    $("#case-select").innerHTML = options;
    $("#cmp-case").innerHTML = options;
    $("#case-select").value = "H01";
    $("#cmp-case").value = "M03";

    await refreshRuns();
    if (state.runs.length) await loadRun(state.runs[0].id);
    updatePlaygroundControls();
    await openFromHash();
  } catch (err) {
    toast(`Không tải được dữ liệu: ${err.message}`);
  }
}

// Deep links for demos: #explorer/H01, #compare/M03, #gate, #playground/A01
async function openFromHash() {
  const [tab, caseId] = location.hash.replace(/^#/, "").split("/");
  const known = caseId && caseById(caseId.toUpperCase()) ? caseId.toUpperCase() : null;
  if (tab === "explorer") {
    $(".tab[data-tab='explorer']").click();
    if (known) await openExplorerCase(known);
  } else if (tab === "compare") {
    if (known) $("#cmp-case").value = known;
    $(".tab[data-tab='compare']").click();
  } else if (tab === "gate") {
    $(".tab[data-tab='gate']").click();
    await runGate($("#gate-btn"));
  } else {
    if (known) { $("#case-select").value = known; updatePlaygroundControls(); }
    if (state.runs.length) await runPlayground();
  }
}

init();
