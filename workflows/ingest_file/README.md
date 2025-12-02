# Workflow: `ingest_file`

Workflow chịu trách nhiệm xử lý `DataSource` đã được upload, tải PDF từ MinIO, kiểm tra tính toàn vẹn, parse với Marker, xây dựng cây tri thức phân cấp và trả về root node. Mục tiêu là giữ orchestration đơn giản và deterministic để có thể tái sử dụng trong các pipeline khác.

## Luồng xử lý

1. **Load DataSource**
   - Mở session Postgres, load `DataSource` theo `data_source_id`.
   - Kiểm tra:
     - Record tồn tại.
     - Status không phải `"active"` (chưa được ingest).
     - `config` chứa `minio_bucket` và `minio_path`.

2. **Download & Hash verify**
   - Tải PDF từ MinIO theo bucket/path đã lưu.
   - Tính MD5 và so sánh với `DataSource.hash_content`.
   - Dừng workflow nếu hash không khớp.

3. **Marker conversion**
   - `MarkerClient` chuyển đổi PDF sang structured JSON (và images) bằng Marker SDK.
   - Chạy trong thread pool để tránh block event loop.

4. **Build Knowledge Tree**
   - Raw Marker JSON được truyền vào `TextbookParser`.
   - Xây dựng cây `StructureNode` (root STRUCTURE node, các STRUCTURE nodes con, DETAIL nodes cho paragraphs và images).
   - Gán parent IDs và positions qua shared helpers.

5. **Validation**
   - Mọi node trừ root phải có `parent_id` không null.
   - `source_id` được gắn vào mọi node và khớp với `DataSource` id.
   - `level_down` và `level_up` phải >= 0 (detail nodes có thể tạm thời = 0).
   - Trên mỗi path từ root đến detail: `level_down + level_up` phải là hằng số.
   - Detail nodes (PARAGRAPH, IMAGE, TABLE) phải có nội dung không rỗng (text/url/headers).

6. **Calculate Statistics**
   - Count nodes by level
   - Count total nodes và total paragraphs
   - Calculate processing duration
   - Calculate average content length

7. **Metadata update**
   - Khi thành công, set `DataSource.updated_at` = timestamp hiện tại và `status` = `"active"`.
   - Cây tri thức **không được persist** bởi workflow này; được trả về cho caller để xử lý tiếp.

## Cấu hình & phụ thuộc

| Biến môi trường | Vai trò |
|-----------------|---------|
| `DATABASE_URL` | Kết nối Postgres (asyncpg). |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | Thông tin MinIO client. |
| `MINIO_KNOWLEDGE_BUCKET` | Bucket chứa file (default `knowledge-files`). |

Phụ thuộc chính: `loguru`, `postgres_client`, `minio_client`, `marker-python`.

## Chạy thử

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db"
export MINIO_ENDPOINT="http://localhost:9000"
export MINIO_ACCESS_KEY="minio"
export MINIO_SECRET_KEY="minio123"
export MINIO_KNOWLEDGE_BUCKET="knowledge-files"

python -m workflows.ingest_file.run
```

Hoặc gọi trực tiếp:

```python
import asyncio
from uuid import UUID

from workflows.ingest_file import FileIngestionInput, start_ingest

async def main() -> None:
    payload = FileIngestionInput(
        data_source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",  # UUID as string
        enable_ocr=False,
    )

    result = await start_ingest(payload)
    print(result.status)           # "complete" hoặc "failed"
    print(result.data_source)      # DataSource object (status='active' after success)
    print(result.root_node)        # Root StructureNode của cây tri thức
    print(result.statistics)       # Statistics dict (total_nodes, processing_duration, etc.)

asyncio.run(main())
```

Đảm bảo Postgres & MinIO đang chạy, và `DataSource` đã được tạo bởi `upload_file` workflow.
