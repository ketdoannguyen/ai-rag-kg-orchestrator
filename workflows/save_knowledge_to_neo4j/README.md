# Workflow: `save_knowledge_to_neo4j`

Workflow chịu trách nhiệm lưu toàn bộ knowledge tree (output từ `enrich_knowledge_tree`) vào Neo4j graph database, tạo nodes và relationships để hỗ trợ truy vấn và phân tích mối quan hệ giữa các node.

## Luồng xử lý

1. **Validation**
   - Kiểm tra `enriched_tree` phải là `StructureNode` (root node).
   - Xác nhận root node có `node_type=STRUCTURE`.

2. **Recursive Tree Saving**
   - Duyệt tree theo depth-first: lưu parent trước, sau đó lưu children.
   - Mỗi node được lưu với:
     - Labels: `KnowledgeNode` + type-specific label (`Structure`, `Paragraph`, `Image`, `Table`).
     - Properties: `id`, `node_type`, `attributes` (JSON), `metadata` (JSON), `level_down`, `level_up`.
   - Dùng `tenacity` để retry tối đa 3 lần nếu lưu node thất bại.

3. **Relationship Creation**
   - **HAS_SECTION**: Structure → Structure (khi structure node có con là structure node).
   - **HAS_CONTENT**: Structure → Detail (khi structure node có con là paragraph/image/table).
   - **NEXT_SIBLING**: Detail → Detail (chỉ giữa các detail nodes cùng loại, không tạo cho structure nodes).
   - Relationships được tạo sau khi tất cả nodes đã được lưu.

4. **Partial Success Strategy**
   - Nếu một node lưu thất bại sau retries, workflow vẫn tiếp tục với children.
   - Failed nodes được ghi lại trong output để xử lý sau.

5. **Response**
   - `SaveKnowledgeToNeo4jOutput` gồm:
     - `root_node_id`: ID của root node.
     - `total_nodes_saved`: Tổng số nodes đã lưu thành công.
     - `structure_nodes_count`: Số structure nodes.
     - `detail_nodes_count`: Số detail nodes (paragraph/image/table).
     - `relationships_created`: Thống kê số lượng từng loại relationship.
     - `saved_nodes`: Danh sách các nodes đã lưu thành công.
     - `failed_nodes`: Danh sách các nodes lưu thất bại (nếu có).
     - `partial_success`: `True` nếu có nodes thất bại.

## Cấu hình & phụ thuộc

| Biến môi trường | Vai trò |
|-----------------|---------|
| `NEO4J_URI` | Neo4j connection URI (e.g., `bolt://localhost:7687`). |
| `NEO4J_USER` | Neo4j username. |
| `NEO4J_PASSWORD` | Neo4j password. |

Phụ thuộc chính: `loguru`, `tenacity`, `neo4j_client`, `postgres_client`.

## Chạy thử

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

python -m tests.workflows.test_save_knowledge_to_neo4j
```

Hoặc gọi trực tiếp:

```python
import asyncio
from uuid import UUID
from workflows.save_knowledge_to_neo4j import SaveKnowledgeToNeo4jInput, start_save_knowledge_to_neo4j

async def main() -> None:
    # enriched_tree từ output của enrich_knowledge_tree workflow
    payload = SaveKnowledgeToNeo4jInput(
        enriched_tree=enriched_tree,  # StructureNode instance
    )

    result = await start_save_knowledge_to_neo4j(payload)
    print(result)

asyncio.run(main())
```

Đảm bảo Neo4j đang chạy và có thể kết nối.

