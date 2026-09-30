# Bàn giao OCR → chunk → knowledge tree research

**Trạng thái lịch sử (2026-09-28):** đã chạy live OCR/merge, nhưng verification sau đó phát hiện các gap F1–F7 về provenance, accounting, pairing, fidelity, docs và tooling; **chưa được nghiệm thu đầy đủ**. Độ chính xác chưa đo được vì thiếu human gold; tiền chưa tính được vì thiếu bảng giá.

Nguồn kiểm chứng: `.claude/reports/2026-09-28-1529-verify-linear-exploring-token.md`; kế hoạch sửa tiếp: `/home/nkd/.claude/plans/serene-crunching-hopcroft.md`; kết quả sau khi sửa F1–F7 và chạy lại research: `.claude/reports/2026-09-29-0510-f1-f7-close-and-live-research.md` (study JSON trong `_r7logs/f1f7-*`). README của package (`research/ocr_pipeline/README.md`) mô tả CLI; file này phân biệt **cái gì đã chạy thật với cái gì chưa đo được**.

## 0. Cập nhật (2026-09-30): gold viết tay, chỉ còn token, bỏ `marker_json`

Bản sửa tiếp theo **đã xoá** một phần surface mà các mục 1–3 dưới đây ghi lại. Các mục đó
vẫn là bằng chứng lịch sử (append-only) — chúng mô tả những lần chạy đã xảy ra, **không**
mô tả code hiện tại:

* `marker_json` đã bị xoá khỏi code: không còn engine, không còn profile `marker_json_llm`,
  không còn `engine_inputs.marker_json_path`. Các run `*.marker_json*` trong store vẫn đọc
  được như bằng chứng, nhưng không còn đường nào chạy lại chúng.
* Toàn bộ code nói về tiền đã bị xoá: không còn `--pricing`, price snapshot, `cost` trong
  report hay study summary. So sánh chỉ còn **input token và output token**
  (`paired_usage.pairs[].usage` với `attempts` / `prompt_tokens` / `completion_tokens`;
  delta chỉ có khi cả hai phía đều khai, thiếu một phía là `incomplete_pairing`).
* Gold đổi schema thành `{doc_id, pages: [{page_index, text, headings?}]}`, viết tay và
  **coi như đã human check**. Không còn `reference_version`, `review_status`,
  `human_checked`/`assistant_draft`, file annotations, `reference freeze`,
  `reference agreement`. Lệnh thay thế: `gold check --doc D --file F`, trả
  `covered_pages` / `missing_pages` / `out_of_scope_pages` / `complete`.
* Corpus manifest bỏ `annotation_path` và `engine_inputs`. Vì `data/research/**` là
  append-only, `data/research/ocr_pipeline/corpus-10pages.json` vẫn còn hai khoá đó và do
  `extra="forbid"` **không load được nữa** (`CorpusError … ['documents.0.annotation_path',
  'documents.0.engine_inputs']`). Muốn chạy lại thì bỏ hai dòng đó ở một bản sao; **không
  sửa file trong `data/research/`**.
* `--llm-mode fake` và client merge giả đã bị xoá khỏi production (double chuyển sang
  `tests/unit/research/ocr_pipeline/merge_double.py`). Merge chỉ còn `none` hoặc `live`.

**Gate của bản sửa này (2026-09-30):**

* `pytest -q tests/unit/research tests/test_builders.py tests/test_resources.py`: **657 passed**.
* `pytest -q tests/unit`: **1421 passed, 4 failed** (1425 collected). Chạy rộng hơn
  (`tests/unit` + `tests/test_builders.py` + `tests/test_resources.py`): **1505 passed,
  4 failed** (1509 collected). Cả hai đều đúng 4 failure allocator ở
  `tests/unit/workflows/benchmark/test_query_allocator.py` đã có trong baseline (§3).
* Scoped `ruff check` + `ruff format --check` trên `research/ocr_pipeline`,
  `scripts/run_ocr_study.py`, `tests/unit/research`, `tests/test_builders.py`,
  `tests/test_resources.py`: sạch. `make mypy-ocr`: **Success, 23 source files**.
* End-to-end trên `/tmp/ocr-e2e` (root scratch, không phải `data/research/`), chạy lại ngày
  2026-09-30 với `--root /tmp/ocr-e2e/root`:
  * `corpus validate` → 10 trang, `pages_with_text: []`, 4 engine, không gọi mạng.
  * `gold check --doc vl12-10pages --file gold.json` → covered `[0,1,2]`, missing `[3..9]`,
    `complete: false`.
  * `compare --family chunk` trên `vision_direct.r1.live-a3` + `vision_atomic_merge.r1.live-a3`
    → `compatible: true`, delta chunk `-17` (40 → 23), `usage_a` đo được (10 / 18756 / 10352),
    `usage_b` đo được (21 / 32459 / 143796, merge tách riêng 11 / 13703 / 133444), usage delta
    `+11 / +13703 / +133444` — đúng bằng phần merge tự khai.
  * `compare --family ocr` vision ↔ native → `compatible: false`,
    `incompatible_reasons: ["run_b_unsupported_modality"]`, mọi metric `not_measured`
    (`runs_incompatible`), và cặp **không có** khoá `usage` nào — không có số 0 nào để đọc nhầm.
  * `study --families pipeline,ocr,chunk --engines vision,marker,docling,native --dry-run`
    → `executed: false`, 4 extraction + 5 tree, cache 3 hit / 6 miss, tự nêu
    `vision is planned but not permitted: pass --allow-upload` và
    `docling is unavailable: docling is not installed`.

**Điều §2 dưới đây nói đã đổi:** mục 2 ghi CER/WER/heading/boundary là `not_measured` vì
gold khi đó có 0 trang `human_checked`. Gold viết tay hiện phủ 3 trang, nên các metric đó
**đã đo được** (vẫn là mẫu rất nhỏ, không đủ để kết luận chất lượng):

* `text_metrics.runs[*].rates` — CER micro `0.3301` (999/3026 ký tự), WER micro `0.3324`
  (226/680 từ), trên 3 trang có gold.
* `structure_metrics` — `pages: {considered: 3, marked: 3, scored: 1, skipped: 2}`,
  `skipped: {alignment_below_threshold: 2}`; trên trang được chấm: heading F1 `1.0`,
  boundary F1 `1.0` (3/3, 0 FP, 0 FN), `alignment: 0.9623`.
* Hai nhánh của family `chunk` đọc **cùng một** artifact nên CER/WER/F1 giống hệt nhau —
  family này đổi cây, không đổi chữ.

## 1. Đã chạy thật (không phải fake, không phải replay)

| Engine | Bằng chứng | Ghi chú |
| --- | --- | --- |
| `vision` | 10/10 trang, live; artifact `fa1a6ee2…` (r1), `4e995bed…` (r2) | Endpoint vision đã được người dùng cho phép |
| `docling` | 10/10 trang; artifact `6525897a…`; cpu 417,99 s, rss_peak 3,05 GB thập phân (~2,84 GiB) | Đã cài thật trong môi trường engine riêng; RSS là sampled estimate |
| `native` | 10/10 trang → `unsupported_modality` | Đúng: PDF là scan, không có text layer |
| `marker` | 1/1 trang; artifact `8c1de0a7…`; rss_peak 7,03 GB thập phân | Run trực tiếp 2 trang bị OOM-kill gần 7,43 GB theo kernel log, **không phải** một stored two-page run |
| `marker_json` | 10 trang / 183 block từ dump đông lạnh `marker-original.json` | Replay, **không** phải một lần Marker extraction |

**Detail merge live (có gọi model thật):**

- `vl12-10pages.vision_atomic_merge.r1.live-a3` — 11 section merged, 0 fallback, 13 703 prompt + 133 444 completion token.
- `vl12-10pages.vision_atomic_merge.r2.live-a2` — tương tự cho repeat 2.
- `vl12-10pages.marker_json_llm.r1.live-a2` — 13 section merged, 0 fallback, detail 156 → 24, 14 526 prompt + 151 276 completion token, 541,6 s.
- `vl12-10pages.marker_json_llm.r2.live-a2` — 141 807 completion token, 518,8 s.

Các run `.live` / `.live-a2` **cũ hơn** của cùng profile có `llm_calls=0` và `sections_fallback` bằng đúng số section: đó là các bản ghi do code trước bản vá (gateway chết, và code cũ đếm fallback thành merged). Chúng vẫn nằm trong store vì store là append-only; resume đã được sửa để **đi vòng qua** chúng (đòi hỏi `sections_merged > sections_fallback`), nên lần study gần nhất không tái sử dụng chúng.

**Study end-to-end:** `data/research/ocr_pipeline/_r7logs/r7-study-oc2.json`, exit 0 — 8/8 extraction ok, 12/12 tree ok, `skipped: []`, 3/3 comparison (`pipeline`, `ocr`, `chunk`). Bất biến của family `chunk` đúng trên dữ liệu thật: hai nhánh của mỗi cặp đọc **cùng một digest** (`fa1a6ee2…` cho r1, `4e995bed…` cho r2), `compatible: true`, delta đo được −17 và −14 chunk.

## 2. Cái KHÔNG đo được, và vì sao

| Thứ | Trạng thái | Nguyên nhân |
| --- | --- | --- |
| CER / WER / heading / boundary | `not_measured` | Gold có 10 trang nhưng **0 trang `human_checked`** (`gate: unlabeled_only`). Không được tự sinh gold từ output của engine |
| Cost | `not_measured` (`model_not_priced`, `no_price_snapshot`) | Chưa có bảng giá nào được đưa vào. Manifest có token thật nhưng thiếu giá thì không quy ra tiền được |
| `marker_llm` merge | `not_measured` (`merge_rebuilt_nothing`) | Artifact Marker thật chỉ 1 trang → cả tài liệu có **1 detail** → merge không có gì để gộp, 4/4 section bị skip, 0 call. Đây là thuộc tính của input, không phải lỗi |
| Resources của family `pipeline` (report cũ) | `{}` | Tree replay không đo engine mới; report cũ **chưa join** resources của extraction run nguồn vào tree arm. Không kết luận extraction không dùng RAM/CPU. Bản sửa tiếp đã join theo run ID (`resource_run_id` trên row, stages/resources giữ nguyên khoá run của extraction) |
| So sánh `marker_llm` với `vision_direct` | không cặp được | Marker chỉ chạy 1 trang, Vision chạy 10 trang — khác page selection |

**Đọc số cho đúng:** `refine.llm_model` là tên model **được yêu cầu**, không phải bằng chứng model đã trả lời. Dù đặt `BENCHMARK_LLM_MODEL` khớp cấu hình launcher, record cũ không có served-model attribution độc lập và không thể định giá hồi tố từ tên requested. Các run cũ ghi `claude-sonnet-4.6` trong khi endpoint phục vụ model khác. Family `pipeline` cũ ghép all-pairs, không theo fixed pair `vision_direct ↔ marker_llm`; frozen `marker_json_llm` phải được gắn nhãn substitute do giới hạn RAM, không phải Marker extraction mới.

## 3. Gate lịch sử (2026-09-28; không đại diện cho bản sửa tiếp)

- Focused gate theo verification: **916 passed, 1 skipped**.
- `pytest -q tests/unit/research/ocr_pipeline/test_study_drop_in.py`: **16 passed**.
- `pytest -q tests/unit`: **1498 passed, 4 failed**; 4 failure allocator ở `tests/unit/workflows/benchmark/test_query_allocator.py` có trong baseline.
- Scoped Ruff check/format: sạch; root Ruff còn **140 errors ngoài phạm vi OCR**.
- Scoped mypy lúc đó còn **43 diagnostics**; root mypy vướng duplicate `samples`. `python -m build` khi đó thiếu frontend nên chưa xác nhận build; lần sửa tiếp đã build root/gateway wheel offline bằng `uv build --offline --no-build-isolation` và kiểm nội dung gateway. Không coi gate lịch sử là gate của bản code mới.
- Privacy gate trên 3 comparison cũ: không thấy secret/transcript dài trong report; chỉ chứng minh những file đã kiểm.

## 4. Lệnh an toàn để bắt đầu

```bash
git status --short && git diff --check
.venv/bin/python scripts/run_ocr_study.py --help
.venv/bin/pytest -q tests/unit/research/ocr_pipeline
.venv/bin/python scripts/run_ocr_study.py corpus validate --corpus data/research/ocr_pipeline/corpus-10pages.json
.venv/bin/python scripts/run_ocr_study.py study --corpus data/research/ocr_pipeline/corpus-10pages.json --dry-run
```

`--dry-run` chỉ đọc store và in kế hoạch, **không** gọi model. Các lệnh dưới đây **tốn phí thật hoặc tải model nặng**, chỉ chạy khi có đồng ý rõ ràng: `extract --engine vision|docling|marker`, `run --llm-mode live`, `study` không có `--dry-run`, `pip install` engine.

Muốn cấu hình một lần merge live tới gateway đã cho phép, trỏ chat gateway sang endpoint vision qua biến môi trường tạm (không ghi `.env`). Tên model cấu hình **không** chứng minh model trả về; chỉ response metadata mới là attribution độc lập:

```bash
OC_CHAT_MODEL=<model> PYTHONPATH=$PWD .venv/bin/python /tmp/oc_exec.py \
  scripts/run_ocr_study.py run --corpus <corpus> --profile <profile> --llm-mode live --allow-merge-upload
```

`/tmp/oc_exec.py` đọc credential từ settings trong tiến trình, **không in ra và không ghi vào `.env`**, nên thay đổi chỉ có tác dụng cho đúng lệnh nó chạy. Nó không nằm trong repo.

## 5. Ràng buộc không được vi phạm

- Không commit / push / PR. Không in `.env`, response của provider, header API/session, hay transcript sách giáo khoa vào log hoặc file được track.
- Không commit PDF, JSON gold, ảnh render, bản chép SGK.
- `data/research/**` là bằng chứng **append-only**: không sửa, không xoá, không chạy lại đè.
- Chỉ `research/` được import `workflows/`; không bao giờ ngược lại. Runtime không được import `research/`.
- Không ghi ArangoDB, MinIO hoặc DataSource.
- Giữ nguyên các file đang dirty của người dùng (`.env.example`, `.gitignore`, `Makefile`, docker, tests/runtime liên quan).
- Marker/Docling là môi trường engine **tuỳ chọn có ghim version**; chúng không được trở thành dependency mặc định của app.
