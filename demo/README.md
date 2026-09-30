# OrbitTech Eval Lab — Demo UI

Giao diện demo cho bài lab Day 14. Backend dùng lại trực tiếp
`domain_assistant.py` (BM25 + prompt + OpenAI) và `template.py` (metrics,
`BenchmarkRunner`, `FailureAnalyzer`), nên số liệu trên UI chính là số của
evaluation core. Chỉ dùng Python standard library và HTML/CSS/JS thuần —
không thêm dependency ngoài `requirements.txt`.

## Chạy

```bash
# từ thư mục gốc của repo, đã activate .venv
python demo/server.py            # mở http://127.0.0.1:8014
python demo/server.py --port 9000
```

Lần chạy đầu, server đăng ký `artifacts/actual_answers.json` thành
`artifacts/runs/run-001` (baseline). Không có `OPENAI_API_KEY` thì mọi thứ vẫn
chạy, trừ việc sinh answer live (Playground chỉ hiện retrieval).

Deep link hữu ích khi demo:

| Link | Nội dung |
|---|---|
| `/#playground/H01` | Answer đã lưu + đánh giá của H01 |
| `/#explorer/H01` | H01: rule 21 ngày ở chunk #1 nhưng model trả 45 ngày |
| `/#explorer/A01` | A01: BM25 khớp nhầm `stocks → stock` |
| `/#compare/M03` | Reranking làm Context Precision của M03 giảm 1.000 → 0.333 |
| `/#gate` | Quality Gate giữa run đầu và run mới nhất |

## Các tab

- **Playground** — chọn case golden hoặc nhập câu hỏi mới. Chế độ *Answer đã
  lưu* đọc trace từ run (không gọi API); *Chạy live* retrieve + sinh answer với
  tuỳ chọn `top_k`, rerank và prompt version. Chunks hiển thị thứ hạng, nguồn,
  BM25 score, các term khớp (highlight) và gold evidence (gạch chân).
- **Đánh giá** — với case golden: Expected cạnh Actual, Faithfulness /
  Relevance / Completeness / Overall / failure type, Context Recall / Precision.
  Bấm nhãn FAIL để mở phần giải thích: phân tích từ `reflection.md`,
  `find_root_cause()`, gold evidence có được retrieve không, kiểm tra số liệu.
- **Câu hỏi tự nhập** — không có expected answer / gold context nên **không
  chấm điểm**. Chỉ khi người dùng nhập reference answer mới chấm, và UI ghi rõ
  Faithfulness lúc đó được đo với retrieved chunks.
- **Failure Explorer** — lọc theo difficulty, failure type, pass/fail; mở từng
  case để xem trace và giải thích.
- **So sánh** — bật/tắt `rerank_by_overlap()` trên cùng tập chunk (Recall giữ
  nguyên, Precision đổi), chạy cho cả 20 case; hoặc so hai run theo từng case
  (answer, pass/fail, thứ tự chunk).
- **Quality Gate** — đánh giá lại hai run đã lưu bằng `run_regression()`:
  BLOCK khi Faithfulness/Relevance/Completeness giảm > 0.05 hoặc case
  Hard/Adversarial chuyển pass→fail; Recall/Precision/pass rate chỉ cảnh báo;
  liệt kê case Hard/Adversarial cần review. Có thể sinh run mới (prompt 1.0
  hoặc 1.1 thử nghiệm) ngay trên UI.

## Prompt 1.1 (thử nghiệm)

Prompt 1.1 thêm các quy tắc được đề xuất trong `reflection.md` (xác định policy
version theo ngày, xử lý "the longer of", scope rules, một kết luận cuối). Nó
được chèn bằng cách bọc generator trong `demo/server.py`; `domain_assistant.py`
và baseline không thay đổi.

## File

```text
demo/
├── server.py          # HTTP server + JSON API (stdlib)
├── case_notes.json    # phân tích từng failure (viết cho run-001)
└── static/            # index.html, styles.css, app.js
artifacts/runs/run-NNN/  # actual_answers.json, benchmark_results.json, meta.json
```
