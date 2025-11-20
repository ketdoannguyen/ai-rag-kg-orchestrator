from parser_client import ParserClient
from storage_client import StorageClient


def ingest_page(
    page_id: str,
    html: str,
    parser: ParserClient,
    storage: StorageClient,
) -> dict:
    paragraphs = parser.parse_html(html)
    storage.ensure_bucket("facebook")
    storage.upload_bytes(
        bucket_name="facebook",
        object_name=f"{page_id}.html",
        data=html,
        content_type="text/html",
    )
    return {"page_id": page_id, "paragraph_count": len(paragraphs)}
