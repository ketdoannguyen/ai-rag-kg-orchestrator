# Workflow: `upload_file`

Workflow chịu trách nhiệm nhận tài liệu từ người dùng, kiểm tra an toàn rồi lưu vào MinIO và Postgres để phục vụ các pipeline tiếp theo.

## Luồng xử lý

1. **Validation**
   - Kiểm tra extension nằm trong whitelist `.pdf/.docx/.txt/.xlsx/.csv/.pptx`.
   - Giới hạn kích thước ≤ 100MB và không rỗng.
   - Khớp chữ ký nội dung (PDF header, cấu trúc Office ZIP, file text không chứa byte NULL).
   - (Tuỳ chọn) scan malware nếu cấu hình `antivirus_scanner`.

2. **Hash & Duplicate guard**
   - Tính MD5 fingerprint của toàn bộ file.
   - Tra bảng `data_sources.hash_content`. Nếu trùng:
     - `fail_on_duplicate=True` → ném `DuplicateFileError`.
     - Ngược lại → trả về record sẵn có + thông tin MinIO mà không upload lại.

3. **Metadata & Upload**
   - Phát sinh `file_id = uuid4()` và tạo object key `{created_by}/{file_id}/{filename}`.
   - Upload lên bucket `settings.MINIO_KNOWLEDGE_BUCKET` (fallback `MINIO_BUCKET`) qua helper `upload_file`.
   - Đính metadata (`Content-MD5`, `X-Upload-User`, `original-filename`).
   - Dùng `tenacity` để retry upload tối đa 3 lần với backoff luỹ thừa và log chi tiết bằng Loguru.

4. **Persist DataSource**
   - Dữ liệu lưu vào bảng `data_sources` qua `postgres_client.handlers.SystemAdminHandler`.
   - `config` chứa bucket, path, size, mime type, `file_id`, `original_filename` + `additional_config`.
   - Status mặc định `processing`.
   - Nếu insert lỗi → xoá object khỏi MinIO và báo `DatabaseError`.

5. **Response**
   - `FileUploadResult` gồm:
     - `data_source`: record vừa tạo (hoặc record trùng lặp).
     - `minio_info`: bucket/path/hash + presigned download URL (hết hạn sau 1h nếu MinIO hỗ trợ).
     - `duplicate`: bool.

## Cấu hình & phụ thuộc

| Biến môi trường | Vai trò |
|-----------------|---------|
| `DATABASE_URL` | Kết nối Postgres (asyncpg). |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | Thông tin MinIO client. |
| `MINIO_KNOWLEDGE_BUCKET` | Bucket đích (default `knowledge-files`). |
| `MINIO_BUCKET` | Bucket dự phòng nếu không đặt biến trên. |

Phụ thuộc chính: `loguru`, `tenacity`, `postgres_client`, `minio_client`.

## Chạy thử

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db"
export MINIO_ENDPOINT="http://localhost:9000"
export MINIO_ACCESS_KEY="minio"
export MINIO_SECRET_KEY="minio123"
export MINIO_KNOWLEDGE_BUCKET="knowledge-files"

python -m workflows.upload_file.run
```

Hoặc gọi trực tiếp:

```python
import asyncio
from pathlib import Path
from uuid import UUID

from workflows.upload_file import FileUploadInput, start_upload_file

async def main() -> None:
    payload = FileUploadInput(
        file=Path("data/input/SGK-Vat-li-12-KNTT-0-10.pdf").read_bytes(),
        filename="SGK-Vat-li-12-KNTT-0-10.pdf",
        source_type="FILE",
        created_by=UUID("550e8400-e29b-41d4-a716-446655440000"),
    )

    result = await start_upload_file(payload)
    print(result)

asyncio.run(main())
```

Đảm bảo Postgres & MinIO đang chạy, bucket tồn tại hoặc để workflow tự tạo.
