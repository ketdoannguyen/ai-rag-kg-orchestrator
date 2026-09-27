# Workflow: `ingest_file`

Workflow xử lý một `DataSource` đã upload: tải PDF từ MinIO, kiểm tra toàn vẹn, **OCR từng trang bằng LLM vision**, chuẩn hóa kết quả thành cây tri thức phân cấp rồi trả về root node. Cây chỉ tồn tại in-memory — workflow không persist node vào ArangoDB.

Engine duy nhất là vision LLM. Không còn Marker, không còn `TextbookParser`, không có fallback engine.

## Luồng xử lý

1. **Load DataSource** — load theo `data_source_id`; kiểm tra record tồn tại, status không phải `"active"`, và `config` có `minio_bucket`/`minio_path`.
2. **Download & Hash verify** — tải file từ MinIO, tính MD5 và so với `DataSource.hash_content`; lệch hash thì dừng.
3. **Render & OCR từng trang** — `render_pdf_pages` render PDF bằng `pypdfium2` ở `VISION_DPI`, mỗi lần một trang; mỗi trang được gửi tới vision endpoint kèm `OCR_STATE` (chuỗi heading đang mở + đuôi đoạn đang mở) và chỉ được chấp nhận khi HTTP 200, `finish_reason == "stop"`, JSON parse được và schema hợp lệ.
4. **Build Knowledge Tree** — `VisionDocumentParser` chuẩn hóa block, gộp block cùng `group` thành chunk rồi dựng cây `StructureNode`/`ParagraphNode`/`TableNode`; heading stack sống xuyên trang. Sau đó gán parent IDs và positions bằng shared helpers.
5. **Validation** — mọi node trừ root phải có `parent_id` khớp id của cha; `source_id` gắn vào mọi node và khớp `DataSource`; `level` >= 0; paragraph phải có text không rỗng, table phải có rows hoặc headers.
6. **Statistics** — số node theo level, total nodes/paragraphs, thời lượng, độ dài nội dung trung bình, kèm số trang, số lần retry, token và counter chuẩn hóa.
7. **Metadata update** — thành công: `updated_at` = hiện tại và `status` = `"active"`. Thất bại (khi đã load được DataSource): `status` = `"error"`, trừ khi source đang `"active"` thì giữ nguyên.

Một trang lỗi sau khi hết retry làm **toàn bộ workflow FAILED** — không có trang nào bị bỏ qua, vì thiếu trang sẽ để lại lỗ hổng trong cây phân cấp.

## Semantic normalization

Model chỉ đề xuất block có kiểu; code quyết định block nào thành section. Các rule tập trung trong `vision_parser.py`:

- `EM ĐÃ HỌC`, `EM CÓ THỂ`, `EM CÓ BIẾT` luôn là callout, không bao giờ vào heading stack.
- Heading chứa dấu `?`, hoặc không có marker mà mở đầu bằng động từ hướng dẫn (`Nêu`, `Trình bày`, `Giải thích`, ...), bị hạ thành question. Heading **có** marker được tin là cấu trúc thật (`b) Giải thích sự nóng chảy ...` là tiêu đề mục con).
- `depth` của model là cấp tương đối: node level = `depth + 1`, chặn trong `[2, 8]`; nhảy quá một cấp so với cha đang mở thì bị clamp và ghi counter.
- `table` thành `TableNode` đứng riêng (không bao giờ bị gộp vào group) với `extracted_text` và parser pipe/tab tối giản; `figure`/`equation` giữ nguyên chữ nhìn thấy dưới dạng paragraph (text-first, chưa có crop/bbox nên không tạo `ImageNode`).
- Nội dung không bị sửa chính tả, đổi hoa/thường hay dịch. Counter audit: `callouts_reclassified`, `questions_reclassified`, `depth_clamped`, `duplicate_headings`, `tables_fallback`.

## Gộp chunk theo `group`

Model **gán nhãn**, code **nối** — model không bao giờ viết lại chữ khi gộp.

- Mỗi block có `group: int`. Các block **liền kề** cùng số thuộc một ý và được nối bằng `\n` thành **một** `ParagraphNode`; đổi số là mở chunk mới. Heading luôn cắt group; `table` luôn đứng riêng.
- Model chỉ cần so với block liền trước, không phải giữ id nhất quán toàn cục — thứ tự đọc luôn được bảo toàn và không có id treo.
- `group` là field **bắt buộc**: thiếu hoặc âm thì schema mismatch và cả trang bị retry (fail-closed), thay vì âm thầm rơi về chunk theo từng block.
- Nối xuyên trang cần **hai** điều kiện, không chỉ một cờ. `continues_previous: bool` ở **cấp trang** là nửa đầu: model khai báo block đầu trang nối tiếp ý đang mở ở cuối trang trước. Nửa sau nằm ở nhãn: `group` là nhãn **cục bộ trong một trang**, model đánh số lại từ đầu mỗi trang, nên code cấp thêm `open_group` (số group đang mở) vào `OCR_STATE` và prompt buộc block nội dung đầu tiên của trang phải dùng **đúng** số đó. Chỉ khi nhãn trùng thì hai trang mới thật sự thành một chunk; khai báo suông mà đánh số lại thì code cắt, không ghép bừa hai ý khác nhau.
- `OCR_STATE` còn gửi `open_group_tail` — 200 ký tự cuối của group đang mở — để model thấy mình đang nối vào cái gì, chứ không chỉ nhận một con số.
- Heading chèn giữa luôn thắng: trang nối tiếp mà mở đầu bằng heading thì `_flush()` cắt group trước đó, nên một chunk không bao giờ vắt qua ranh giới mục.
- Chunk bị cắt khi vượt `MAX_CHUNK_SIZE` (`config/chunking.py`, = 1024) — cùng trần mà `enrich_knowledge_tree` đang nhắm.
- Lưới an toàn `FRAGMENT_CHARS = 150`: hai chunk liền kề cùng section mà đều dưới 150 ký tự thì được nối, tránh mảnh vụn như `"Đèn cồn (3)."`.
- Counter audit: `groups_created`, `groups_split`, `short_groups_merged` (lưới an toàn), và cặp `cross_page_continues` / `cross_page_joins` — số trang **khai báo** nối tiếp so với số lần **nối thật qua một ranh giới trang**. Hai số này lệch nhau là bình thường: trang khai `true` nhưng mở đầu bằng heading, hoặc đánh số lại, thì không nối. Một chunk vắt ba trang sẽ làm `cross_page_joins` tăng hai lần.

Vì producer đã ra chunk đúng cỡ, luồng PDF tắt bước merge của enrich: `scripts/run_pdf_pipeline.py` truyền `RefineConfig(refine_details=False)`. Enrich khi đó chỉ còn sinh `summary` cho structure node. `ingest_web` không gộp chunk nên vẫn để `refine_details=True`.

## Cấu hình & phụ thuộc

| Biến môi trường | Vai trò |
|-----------------|---------|
| `ARANGO_URL`, `ARANGO_DATABASE`, `ARANGO_USER`, `ARANGO_PASSWORD` | Kết nối ArangoDB. |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | Thông tin MinIO client. |
| `INGEST_FILE_WORKDIR` | Thư mục tạm cho file tải về (default `data/ingest_file`). |
| `VISION_API_BASE` | Endpoint OpenAI-compatible `/chat/completions`. |
| `VISION_API_KEY` | Bearer token. Chỉ đọc từ environment, không log/serialize. |
| `VISION_MODEL` | Tên model vision. |
| `VISION_ALLOWED_HOSTS` | Danh sách host (phân tách bởi dấu phẩy) được phép nhận key. Rỗng = không giới hạn. |
| `VISION_HEADERS` | Header phụ dạng dict. `Authorization` trong đây bị bỏ qua. opencode.ai **bắt buộc** có `x-opencode-session`, thiếu là HTTP 400 `MissingSessionID`. |
| `VISION_IMAGE_DETAIL`, `VISION_DPI`, `VISION_TEMPERATURE`, `VISION_MAX_TOKENS` | Tham số request/render. |
| `VISION_REASONING_EFFORT` | Gợi ý ngân sách suy luận. Model là reasoning model nên nó tiêu `VISION_MAX_TOKENS` vào suy luận ẩn *trước khi* viết JSON, vì vậy trần thấp trả về câu trả lời **rỗng** (`finish_reason=length`, 0 ký tự) chứ không phải câu trả lời ngắn. `none` tắt hẳn suy luận — đo được khoảng 8 lần ít token hơn và vẫn giữ nguyên độ trung thực; `low`/`medium`/`high` để mua lại suy luận; để rỗng thì không gửi tham số. |
| `VISION_MAX_BLOCKS_PER_PAGE` | Trần số block mỗi trang; vượt thì coi là lỗi và retry. |
| `VISION_PAGE_OFFSET` | Số trang in = số trang PDF + offset. |

`VISION_API_BASE` trỏ host nào thì phải thêm host đó vào `VISION_ALLOWED_HOSTS`, nếu không client sẽ từ chối gửi key. Client không bao giờ gửi key tới host ngoài allowlist.

Phụ thuộc chính: `httpx`, `pypdfium2`, `pillow`, `loguru`, `arango_client`, `minio_client`.

Retry (`max_retries` trong input, tối đa 5) áp dụng cho timeout, lỗi transport, HTTP 429/5xx, response bị cắt, JSON hỏng và schema mismatch. Các lỗi 400/401/403/404/422, thiếu key hoặc host ngoài allowlist fail ngay, không retry.

## Chạy thử

```bash
export ARANGO_URL="http://localhost:8529"
export MINIO_ENDPOINT="http://localhost:9000"
export VISION_API_BASE="https://<host>/v1/chat/completions"
export VISION_API_KEY="..."
export VISION_MODEL="<vision-model>"
export VISION_ALLOWED_HOSTS="<host>"

python scripts/run_ingest_file.py <data_source_id> --output data/ingest_result.json
```

Hoặc gọi trực tiếp (hàm đồng bộ, tự gọi `asyncio.run` bên trong):

```python
from workflows.ingest_file import FileIngestionInput, start_ingest

result = start_ingest(
    FileIngestionInput(data_source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890")
)

print(result["status"])       # "complete" hoặc "failed"
print(result["data_source"])  # DataSource (status='active' sau khi thành công)
print(result["root_node"])    # root node của cây tri thức đã serialize
print(result["statistics"])   # total_nodes, pages_processed, ocr_attempts, ...
```

DataSource phải được tạo trước bởi workflow `upload_file`, và MinIO phải đang chạy.

## Giới hạn đã biết

- Chưa có bbox/crop hình: figure chỉ giữ caption và chữ nhìn thấy, không upload ảnh, không tạo `ImageNode`.
- OCR tuần tự theo trang vì hierarchy phụ thuộc state của trang trước.
- Không cache/resume giữa chừng; chưa lưu raw OCR lâu dài.
