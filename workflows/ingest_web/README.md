# Workflow: `ingest_web`

Workflow chịu trách nhiệm crawl các trang web, trích xuất nội dung, xây dựng knowledge tree và lưu vào Postgres để phục vụ các pipeline tiếp theo.

## Luồng xử lý

1. **Validate URLs**
   - Kiểm tra format hợp lệ (regex: `^https?://`).
   - Kiểm tra domain có accessible không (DNS lookup).
   - Kiểm tra robots.txt nếu `respect_robots_txt=true` trong `crawl_config`.
   - Parse robots.txt: `https://example.com/robots.txt`.
   - Kiểm tra URL có trong disallow list không.
   - Filter out invalid URLs.
   - Nếu tất cả URLs invalid: throw error và stop.

2. **Create DataSource - Tạo nguồn web (cho mỗi URL)**
   - Tạo DataSource object với:
     - `id`: UUID mới
     - `name`: Tạm thời là URL, sẽ được cập nhật sau khi crawl (từ title hoặc heading đầu tiên)
     - `type`: `SourceType.WEB`
     - `config`: Dict chứa URL và crawl config

       ```json
       {
         "urls": ["https://docs.example.com/..."],
         "crawl_config": {...},
         "crawl_initiated_at": "2025-12-01T10:00:00Z"
       }
       ```
     - `hash_content`: `null` (web content không có hash cố định)
     - `status`: `"processing"`
     - `created_by`: User ID
     - Timestamps
   - Save DataSource vào database.

3. **Crawl URLs - Thu thập dữ liệu**
   - Sử dụng `crawl4ai` để crawl từng URL.
   - Trích xuất HTML và markdown từ kết quả crawl.
   - Sử dụng `ContentExtractor` để loại bỏ navigation, ads, boilerplate.

4. **Build Knowledge Tree - Tạo cây phân cấp**
   - Convert markdown thành hierarchical structure sử dụng `MarkdownToJSONConverter`.
   - Tạo root node (CORPUS) với metadata (url, crawled_at, etc.).
   - Parse headings và paragraphs thành `StructureNode` và `ParagraphNode`.
   - Set `parent_id` và `position` cho tất cả nodes.

5. **Validate Nodes - Kiểm tra**
   - Validate tất cả nodes:
     - Check `physical_level` rules (level_down >= 0, level_up >= 0).
     - Check `parent_id` references (root có parent_id=None, non-root phải có parent_id).
     - Check `chunk_text` không empty cho paragraph nodes.
     - Check `level_down + level_up` constant trên mỗi path từ root đến detail.
   - Gán `source_id` cho tất cả nodes.

6. **Calculate Statistics - Thống kê**
   - Count nodes by level.
   - Count total nodes và total paragraphs.
   - Calculate crawl duration.
   - Calculate average content length.

7. **Update DataSource**
   - Cập nhật `name` từ title/heading đầu tiên của content.
   - Cập nhật `config` với statistics và crawl metadata.
   - Lưu vào database.

8. **Response**
   - `WebIngestionResult` gồm:
     - `results`: List các `WebIngestionSingleResult` cho từng URL.
   - Mỗi `WebIngestionSingleResult` gồm:
     - `data_source`: DataSource record vừa tạo/cập nhật.
     - `root_node`: CORPUS node (root) của knowledge tree.
     - `statistics`: Thống kê crawling (nodes_by_level, total_nodes, crawl_duration, etc.).

## Cấu hình & phụ thuộc

| Biến môi trường | Vai trò |
|-----------------|---------|
| `DATABASE_URL` | Kết nối Postgres (asyncpg). |

Phụ thuộc chính: `loguru`, `tenacity`, `postgres_client`, `crawl4ai`, `aiohttp`.

## Input

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `urls` | `List[str]` | ✅ | Danh sách URLs cần crawl | `["https://example.com/page1", ...]` |
| `created_by` | `UUID` | ✅ | User ID | `"550e8400-e29b-41d4-a716-446655440000"` |
| `crawl_config` | `Dict` | ❌ | Cấu hình crawling | `{"respect_robots_txt": true, "max_depth": 2}` |
| `max_url_retries` | `int` | ❌ | Số lần retry tối đa (default: 3) | `3` |
| `per_url_timeout_seconds` | `float` | ❌ | Timeout cho mỗi URL (default: 90.0) | `90.0` |
| `max_concurrent_urls` | `int` | ❌ | Số URL crawl đồng thời (default: 4) | `4` |

## Output

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `results` | `List[WebIngestionSingleResult]` | Danh sách kết quả cho từng URL | Xem bên dưới |

Mỗi `WebIngestionSingleResult`:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `WebIngestionStatus` | Status của việc crawl URL (`SUCCESS`, `ERROR`, `SKIPPED`) |
| `data_source` | `DataSource \| None` | DataSource cho web (status='processing' hoặc 'error'). None nếu status=SKIPPED |
| `root_node` | `KnowledgeNode \| None` | CORPUS node (root) của knowledge tree. None nếu status=ERROR hoặc SKIPPED |
| `statistics` | `Dict \| None` | Thống kê crawling (nodes_by_level, total_nodes, crawl_duration, etc.). None nếu status=ERROR hoặc SKIPPED |
| `error_message` | `str \| None` | Thông báo lỗi nếu status=ERROR hoặc SKIPPED |

## Chạy thử

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db"

python -m workflows.ingest_web.run
```

Hoặc gọi trực tiếp:

```python
import asyncio
from uuid import UUID

from workflows.ingest_web import WebIngestionInput, start_ingest_web

async def main() -> None:
    payload = WebIngestionInput(
        urls=["https://example.com/page1", "https://example.com/page2"],
        created_by=UUID("550e8400-e29b-41d4-a716-446655440000"),
        crawl_config={"respect_robots_txt": True},
    )

    result = await start_ingest_web(payload)
    print(result)

asyncio.run(main())
```

## Lưu ý quan trọng

⚠️ **Status = "processing"**: Cần chạy workflow 3 (embed_nodes) sau đó để hoàn tất ingestion.

⚠️ **Respect robots.txt**: Mặc định workflow sẽ tuân thủ quy tắc crawling của website. Set `respect_robots_txt: false` trong `crawl_config` để bỏ qua.

⚠️ **Rate limiting**: Không crawl quá nhanh, có thể bị block IP. Sử dụng `max_concurrent_urls` để giới hạn.

⚠️ **Error handling**: Một số URLs fail không nên stop toàn bộ workflow. Workflow sẽ tiếp tục xử lý các URL khác.

⚠️ **Retry logic**: Sử dụng `tenacity` để retry tối đa 3 lần với exponential backoff cho mỗi URL.

⚠️ **Mỗi URL = một DataSource**: Mỗi URL sẽ được tạo thành một DataSource riêng biệt với `source_name` được trích xuất từ content của URL đó.

⚠️ **Status tracking**: Mỗi URL sẽ có status riêng:
   - `SUCCESS`: URL được crawl thành công, có data_source, root_node và statistics
   - `ERROR`: URL crawl thất bại sau tất cả retries, có data_source với status='error' nhưng không có root_node
   - `SKIPPED`: URL bị skip do invalid format, DNS failure, hoặc disallowed by robots.txt

⚠️ **Image Handler**: File `image_handler.py` được giữ lại trong codebase để sử dụng trong tương lai nhưng hiện tại chưa được tích hợp vào workflow.
