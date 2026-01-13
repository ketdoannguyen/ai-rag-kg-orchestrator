"""
Visual Data Seeding for ArangoDB Graph Visualization.

This test module creates rich, realistic sample data for visual inspection
via ArangoDB Web UI. It focuses on data seeding rather than assertions.

Usage:
    KEEP_TEST_DB=true pytest tests/libs/arangodb/test_seed_visual_data.py -v -s

Then open ArangoDB Web UI at http://localhost:8529 to visualize the graph.
"""

import random
import uuid
from datetime import datetime, timedelta

import pytest
from faker import Faker

from arango_client.collections import Collections, EdgeCollections, RelationshipTypes
from arango_client.handlers import ChatHandler, KnowledgeHandler
from arango_client.models import (
    ChatMessage,
    ChatSession,
    ParagraphNode,
    StructureNode,
)


# Initialize Vietnamese Faker
fake = Faker("vi_VN")
Faker.seed(42)  # Reproducible fake data
random.seed(42)


# =============================================================================
# Fake Content Generators
# =============================================================================


def generate_fake_embedding(dimension: int = 768) -> list[float]:
    """Generate a random embedding vector with given dimension."""
    return [random.uniform(-1.0, 1.0) for _ in range(dimension)]


def generate_paragraph_content(topic: str, index: int) -> str:
    """Generate realistic Vietnamese paragraph content."""
    templates = [
        f"Theo quy định tại {topic}, thí sinh cần đáp ứng các điều kiện sau: "
        f"(1) {fake.sentence()}; (2) {fake.sentence()}; (3) {fake.sentence()}. "
        f"Ngoài ra, thí sinh phải hoàn thành đầy đủ hồ sơ đăng ký xét tuyển "
        f"theo mẫu quy định của Bộ Giáo dục và Đào tạo.",
        f"Khoản {index} - {topic}: Nhà trường xét tuyển dựa trên các tiêu chí: "
        f"điểm thi THPT, học bạ THPT, và các chứng chỉ quốc tế. "
        f"Điểm xét tuyển = Điểm môn 1 × hệ số 1 + Điểm môn 2 × hệ số 2 + "
        f"Điểm môn 3 × hệ số 3 + Điểm ưu tiên (nếu có).",
        f"Các ngành/chương trình đào tạo áp dụng {topic} này bao gồm: "
        f"Công nghệ thông tin, Khoa học máy tính, Kỹ thuật phần mềm, "
        f"Trí tuệ nhân tạo, An toàn thông tin, và các ngành thuộc khối kỹ thuật. "
        f"Thí sinh lưu ý kiểm tra mã ngành trước khi đăng ký.",
        f"Quy trình xét tuyển theo {topic} được thực hiện qua các bước: "
        f"Bước 1: Đăng ký trực tuyến trên cổng thông tin tuyển sinh. "
        f"Bước 2: Nộp hồ sơ và lệ phí xét tuyển. "
        f"Bước 3: Chờ kết quả xét tuyển. "
        f"Bước 4: Xác nhận nhập học trong thời hạn quy định.",
        f"Thí sinh thuộc diện được hưởng chính sách ưu tiên theo {topic} cần cung cấp "
        f"giấy tờ chứng minh bao gồm: bản sao công chứng giấy khai sinh, "
        f"giấy xác nhận của UBND xã/phường, và các giấy tờ liên quan khác "
        f"theo quy định của Bộ Giáo dục và Đào tạo năm {fake.year()}.",
    ]
    return random.choice(templates)


def generate_chat_conversations() -> list[dict[str, str]]:
    """Generate sample chat Q&A about university admission."""
    return [
        {"role": "user", "content": "Xin chào, cho mình hỏi về phương thức xét tuyển năm 2026?"},
        {
            "role": "assistant",
            "content": "Chào bạn! Năm 2026, trường áp dụng 5 phương thức xét tuyển: "
            "(1) Xét tuyển thẳng theo quy định của Bộ GD&ĐT; "
            "(2) Xét tuyển sớm theo hồ sơ năng lực; "
            "(3) Xét tuyển bằng điểm thi THPT; "
            "(4) Xét tuyển bằng điểm học bạ THPT; "
            "(5) Xét tuyển kết hợp. Bạn quan tâm phương thức nào?",
        },
        {"role": "user", "content": "Mình muốn hỏi về xét tuyển thẳng. Điều kiện như thế nào?"},
        {
            "role": "assistant",
            "content": "Điều kiện xét tuyển thẳng năm 2026 bao gồm:\n"
            "- Thí sinh đạt giải Nhất, Nhì, Ba trong kỳ thi học sinh giỏi quốc gia\n"
            "- Thí sinh đạt giải trong các kỳ thi Olympic quốc tế\n"
            "- Thí sinh có chứng chỉ IELTS ≥ 7.0 hoặc tương đương\n"
            "Bạn có đáp ứng tiêu chí nào không?",
        },
        {"role": "user", "content": "Mình có IELTS 7.5. Vậy cần nộp hồ sơ gì?"},
        {
            "role": "assistant",
            "content": "Với IELTS 7.5, bạn đủ điều kiện xét tuyển thẳng. Hồ sơ cần nộp gồm:\n"
            "1. Đơn xin xét tuyển thẳng (theo mẫu trên website)\n"
            "2. Bản sao công chứng chứng chỉ IELTS còn hiệu lực\n"
            "3. Học bạ THPT (bản sao công chứng)\n"
            "4. CMND/CCCD và 02 ảnh 3x4\n"
            "Thời hạn nộp hồ sơ: trước 30/04/2026.",
        },
        {
            "role": "user",
            "content": "Điểm chuẩn ngành Công nghệ thông tin năm ngoái là bao nhiêu?",
        },
        {
            "role": "assistant",
            "content": "Điểm chuẩn ngành Công nghệ thông tin năm 2025:\n"
            "- Phương thức điểm thi THPT: 26.5 điểm\n"
            "- Phương thức xét học bạ: 27.0 điểm\n"
            "- Phương thức đánh giá năng lực: 850/1000 điểm\n"
            "Lưu ý: Điểm chuẩn 2026 có thể thay đổi tùy số lượng thí sinh.",
        },
        {"role": "user", "content": "Cảm ơn bạn rất nhiều!"},
        {
            "role": "assistant",
            "content": "Không có gì! Chúc bạn đạt kết quả tốt trong kỳ xét tuyển 2026. "
            "Nếu có thắc mắc khác, đừng ngại liên hệ nhé!",
        },
    ]


# =============================================================================
# Test Case: Seed University Knowledge Graph
# =============================================================================


class TestSeedVisualData:
    """Test class for seeding visual inspection data."""

    def test_seed_university_knowledge_graph(self, db, arango_client):
        """
        Seed complete university admission knowledge graph.

        Creates:
        - Hierarchical structure (Root -> Chapters -> Articles -> Paragraphs)
        - Vector embeddings for paragraphs
        - Graph relationships (HAS_SECTION, HAS_CONTENT, RELATED_TO)
        - Chat history with Q&A conversations
        """
        knowledge_handler = KnowledgeHandler(db)
        chat_handler = ChatHandler(db)

        # Track created nodes for relationship building
        created_nodes: dict[str, StructureNode | ParagraphNode] = {}

        print("\n" + "=" * 60)
        print("🌱 SEEDING UNIVERSITY KNOWLEDGE GRAPH")
        print("=" * 60)

        # ---------------------------------------------------------------------
        # STEP 1: Create Root Node
        # ---------------------------------------------------------------------
        print("\n📚 Creating Root Node...")

        root = StructureNode(
            attributes={
                "title": "Quy chế Tuyển sinh Đại học 2026",
                "code": "QC-TS-2026",
                "description": "Quy chế tuyển sinh đại học, cao đẳng hệ chính quy năm 2026. "
                "Ban hành kèm theo Thông tư số XX/2026/TT-BGDĐT ngày 01/01/2026 "
                "của Bộ trưởng Bộ Giáo dục và Đào tạo.",
            },
            metadata={
                "url": "https://tuyensinh.vku.udn.vn/quy-che-2026",
                "start_page": 1,
                "end_page": 50,
            },
            level=1,
            embedding=generate_fake_embedding(768),
            summary="Quy chế tuyển sinh đại học năm 2026 với 5 phương thức xét tuyển.",
        )
        root_created = knowledge_handler.nodes.create_node(root)
        created_nodes["root"] = root_created
        print(f"   ✓ Root: {root_created.attributes.title} (ID: {root_created.id})")

        # ---------------------------------------------------------------------
        # STEP 2: Create Chapter Nodes (Level 2)
        # ---------------------------------------------------------------------
        print("\n📖 Creating Chapter Nodes...")

        chapters_data = [
            {
                "key": "chapter1",
                "title": "Chương I: Quy định chung",
                "code": "CHUONG-I",
                "desc": "Quy định chung về đối tượng, phạm vi áp dụng và các nguyên tắc cơ bản.",
            },
            {
                "key": "chapter2",
                "title": "Chương II: Phương thức xét tuyển",
                "code": "CHUONG-II",
                "desc": "Các phương thức xét tuyển bao gồm xét tuyển thẳng, xét tuyển sớm, "
                "xét điểm thi THPT, xét học bạ và xét tuyển kết hợp.",
            },
            {
                "key": "chapter3",
                "title": "Chương III: Quyền và nghĩa vụ của thí sinh",
                "code": "CHUONG-III",
                "desc": "Quy định về quyền lợi và trách nhiệm của thí sinh trong quá trình "
                "đăng ký và xét tuyển.",
            },
        ]

        for chap in chapters_data:
            node = StructureNode(
                parent_id=root_created.id,
                attributes={
                    "title": chap["title"],
                    "code": chap["code"],
                    "description": chap["desc"],
                },
                metadata={"start_page": random.randint(2, 10), "end_page": random.randint(11, 20)},
                level=2,
                embedding=generate_fake_embedding(768),
                summary=chap["desc"][:100],
            )
            created = knowledge_handler.nodes.create_node(node)
            created_nodes[chap["key"]] = created
            print(f"   ✓ {chap['title']} (ID: {created.id})")

            # Create HAS_SECTION edge from root
            knowledge_handler.nodes.create_has_section_relationship(root_created.id, created.id)

        # ---------------------------------------------------------------------
        # STEP 3: Create Article Nodes (Level 3)
        # ---------------------------------------------------------------------
        print("\n📝 Creating Article Nodes...")

        articles_data = [
            # Chapter 1 articles
            {
                "key": "dieu1",
                "parent": "chapter1",
                "title": "Điều 1: Phạm vi điều chỉnh",
                "code": "DIEU-1",
            },
            {
                "key": "dieu2",
                "parent": "chapter1",
                "title": "Điều 2: Đối tượng áp dụng",
                "code": "DIEU-2",
            },
            {
                "key": "dieu3",
                "parent": "chapter1",
                "title": "Điều 3: Giải thích từ ngữ",
                "code": "DIEU-3",
            },
            # Chapter 2 articles
            {
                "key": "dieu5",
                "parent": "chapter2",
                "title": "Điều 5: Xét tuyển thẳng",
                "code": "DIEU-5",
            },
            {
                "key": "dieu6",
                "parent": "chapter2",
                "title": "Điều 6: Xét tuyển sớm",
                "code": "DIEU-6",
            },
            {
                "key": "dieu10",
                "parent": "chapter2",
                "title": "Điều 10: Xét tuyển kết hợp",
                "code": "DIEU-10",
            },
            {
                "key": "dieu11",
                "parent": "chapter2",
                "title": "Điều 11: Điểm ưu tiên",
                "code": "DIEU-11",
            },
            # Chapter 3 articles
            {
                "key": "dieu15",
                "parent": "chapter3",
                "title": "Điều 15: Quyền của thí sinh",
                "code": "DIEU-15",
            },
            {
                "key": "dieu16",
                "parent": "chapter3",
                "title": "Điều 16: Nghĩa vụ của thí sinh",
                "code": "DIEU-16",
            },
        ]

        for art in articles_data:
            parent_node = created_nodes[art["parent"]]
            node = StructureNode(
                parent_id=parent_node.id,
                attributes={
                    "title": art["title"],
                    "code": art["code"],
                    "description": f"Nội dung chi tiết của {art['title']}.",
                },
                metadata={"start_page": random.randint(5, 30), "end_page": random.randint(31, 45)},
                level=3,
                embedding=generate_fake_embedding(768),
                summary=f"Tóm tắt nội dung {art['title']}",
            )
            created = knowledge_handler.nodes.create_node(node)
            created_nodes[art["key"]] = created
            print(f"   ✓ {art['title']} (ID: {created.id})")

            # Create HAS_SECTION edge from chapter
            knowledge_handler.nodes.create_has_section_relationship(parent_node.id, created.id)

        # ---------------------------------------------------------------------
        # STEP 4: Create Paragraph Nodes (Level 4 - Detail Nodes with Embeddings)
        # ---------------------------------------------------------------------
        print("\n📄 Creating Paragraph Nodes with Embeddings...")

        paragraph_count = 0
        for art_key in ["dieu1", "dieu2", "dieu5", "dieu6", "dieu10", "dieu15"]:
            article = created_nodes[art_key]
            # Create 2-4 paragraphs per article
            for i in range(random.randint(2, 4)):
                content = generate_paragraph_content(article.attributes.title, i + 1)

                para = ParagraphNode(
                    parent_id=article.id,
                    attributes={
                        "text": content,
                        "title": f"Khoản {i + 1}",
                    },
                    metadata={
                        "page": random.randint(1, 50),
                        "bbox": [
                            random.randint(50, 100),
                            random.randint(50, 100),
                            random.randint(400, 500),
                            random.randint(600, 700),
                        ],
                        "position": f"{article.attributes.code}, Khoản {i + 1}",
                    },
                    level=0,
                    embedding=generate_fake_embedding(768),
                    text=content,
                )
                created_para = knowledge_handler.nodes.create_node(para)
                created_nodes[f"{art_key}_para_{i}"] = created_para
                paragraph_count += 1

                # Create HAS_CONTENT edge from article
                knowledge_handler.nodes.create_has_content_relationship(
                    article.id, created_para.id
                )

        print(f"   ✓ Created {paragraph_count} paragraphs with 768-dim embeddings")

        # ---------------------------------------------------------------------
        # STEP 5: Create Cross-Reference Edges (RELATED_TO)
        # ---------------------------------------------------------------------
        print("\n🔗 Creating Cross-Reference Relationships...")

        # Điều 5 (Xét tuyển thẳng) RELATED_TO Điều 10 (Xét tuyển kết hợp)
        edge_coll = db.collection(EdgeCollections.KNOWLEDGE_EDGES)

        cross_refs = [
            ("dieu5", "dieu10", "Tham chiếu phương thức xét tuyển"),
            ("dieu5", "dieu11", "Liên quan điểm ưu tiên"),
            ("dieu6", "dieu10", "Xét tuyển sớm và kết hợp"),
            ("dieu15", "dieu16", "Quyền và nghĩa vụ tương ứng"),
        ]

        for from_key, to_key, desc in cross_refs:
            from_node = created_nodes[from_key]
            to_node = created_nodes[to_key]

            edge_doc = {
                "_from": f"{Collections.KNOWLEDGE_NODES}/{from_node.id}",
                "_to": f"{Collections.KNOWLEDGE_NODES}/{to_node.id}",
                "type": "RELATED_TO",
                "description": desc,
                "created_at": datetime.utcnow().isoformat(),
            }
            edge_coll.insert(edge_doc)
            print(f"   ✓ {from_key} --[RELATED_TO]--> {to_key}: {desc}")

        # ---------------------------------------------------------------------
        # STEP 6: Create Chat History
        # ---------------------------------------------------------------------
        print("\n💬 Creating Chat History...")

        # Create user session
        session = ChatSession(user_identifier="student_2026_001")
        created_session = chat_handler.sessions.create(session)
        print(f"   ✓ Session: {created_session.id}")

        # Create chat messages
        conversations = generate_chat_conversations()
        base_time = datetime.utcnow() - timedelta(hours=1)

        for i, conv in enumerate(conversations):
            msg = ChatMessage(
                session_id=created_session.id,
                role=conv["role"],
                content=conv["content"],
                context_data={
                    "source": "rag" if conv["role"] == "assistant" else None,
                    "latency": random.uniform(0.1, 0.5) if conv["role"] == "assistant" else None,
                },
            )
            # Manually set created_at for proper ordering
            msg_doc = msg.to_document()
            msg_doc["created_at"] = (base_time + timedelta(minutes=i * 2)).isoformat()
            db.collection(Collections.CHAT_MESSAGES).insert(msg_doc)

        print(f"   ✓ Created {len(conversations)} chat messages")

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("✅ SEEDING COMPLETE!")
        print("=" * 60)
        print(f"""
📊 Summary:
   • Root Node: 1
   • Chapter Nodes: {len(chapters_data)}
   • Article Nodes: {len(articles_data)}
   • Paragraph Nodes: {paragraph_count}
   • Cross-Reference Edges: {len(cross_refs)}
   • Chat Sessions: 1
   • Chat Messages: {len(conversations)}

🌐 Open ArangoDB Web UI:
   URL: http://localhost:8529
   Database: test_arango_db
   
📈 Visualize the Graph:
   1. Go to 'Graphs' section
   2. Select 'knowledge_graph'
   3. Click on any root node to expand
        """)

        # Simple assertion to mark test as passed
        assert len(created_nodes) > 0, "Should have created nodes"
        assert paragraph_count > 0, "Should have created paragraphs"
