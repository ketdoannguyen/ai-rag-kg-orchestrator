# Benchmark: Node Similarity Thresholds

## Mục đích

Đánh giá và tìm giá trị tối ưu cho 2 thresholds trong `config/extraction.py`:

1. **EXTRACTION_AUTO_MERGE_THRESHOLD** (hiện tại: 0.72)
   - Similarity >= threshold → AUTO-MERGE (không cần LLM)
   - Rủi ro cao nếu sai: merge 2 nodes khác nhau

2. **EXTRACTION_LLM_THRESHOLD** (hiện tại: 0.50)
   - Similarity >= threshold → Gửi đến LLM để quyết định
   - Similarity < threshold → KEEP SEPARATE (không xử lý)

## Vấn đề hiện tại

- Giá trị 0.72 và 0.50 được đặt cảm tính từ domain khác
- Chưa có benchmark thực nghiệm cho domain **lịch sử Việt Nam**
- Cần test với các cặp node thực tế để tìm thresholds tối ưu

## Decision Tree

```
Similarity Score Decision Tree
==============================

                    ┌─────────────────────────┐
                    │  Calculate Similarity   │
                    │  (Embedding Cosine)     │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   >= AUTO_MERGE         LLM_THRESHOLD -           < LLM_THRESHOLD
   (e.g., 0.72)          AUTO_MERGE                (e.g., < 0.50)
   AUTO-MERGE            (e.g., 0.50-0.72)         KEEP SEPARATE
        │                SEND TO LLM                    │
        ▼                       │                       ▼
┌───────────────┐     ┌─────────────────────┐   ┌───────────────┐
│ Merge nodes   │     │ LLM decides:        │   │ Mark reviewed │
│ (delete one,  │     │ - should_merge      │   │ Keep as is    │
│ keep merged)  │     │ - relation_type     │   │               │
└───────────────┘     │ - confidence        │   └───────────────┘
                      └─────────────────────┘
```

## Cấu trúc

```
benchmarks/node_similarity_threshold/
├── README.md                    # File này
├── samples.py                   # 100 sample node pairs với ground truth
├── test_threshold.py            # Script benchmark
├── query_real_nodes.py          # Query nodes thực từ ArangoDB
├── results/                     # Kết quả benchmark
│   ├── similarities_cache.json  # Cache embeddings
│   └── threshold_analysis.md    # Báo cáo phân tích
└── visualizations/              # Biểu đồ (optional)
```

## Cách chạy

```bash
# 1. (Optional) Query nodes thực từ ArangoDB để tham khảo format
python benchmarks/node_similarity_threshold/query_real_nodes.py

# 2. Chạy benchmark
python benchmarks/node_similarity_threshold/test_threshold.py

# 3. Chạy với verbose
python benchmarks/node_similarity_threshold/test_threshold.py --verbose

# 4. Chạy với custom thresholds
python benchmarks/node_similarity_threshold/test_threshold.py \
    --auto-merge-thresholds 0.68,0.70,0.72,0.75,0.78 \
    --llm-thresholds 0.35,0.40,0.45,0.50,0.55
```

## Sample Categories

### Positive Cases (should_merge=True)
1. **DUPLICATE** - Nội dung giống hệt hoặc gần hệt
2. **PARAPHRASE** - Viết lại cùng ý với từ ngữ khác
3. **SUBSET** - Node A chứa trong Node B hoặc ngược lại

### Negative Cases (should_merge=False)
4. **RELATED_DISTINCT** - Cùng chủ đề nhưng nội dung khác
5. **TEMPORAL_DIFFERENT** - Cùng sự kiện, khác thời điểm
6. **PERSPECTIVE_DIFFERENT** - Cùng sự kiện, khác góc nhìn
7. **CAUSE_EFFECT** - Một là nguyên nhân, một là kết quả

### Boundary Cases (requires LLM)
8. **OVERLAPPING** - Có phần trùng lặp đáng kể
9. **COMPLEMENTARY** - Bổ sung thông tin cho nhau

## Metrics đánh giá

### Cho AUTO_MERGE threshold:
- **Precision**: Trong số auto-merge, bao nhiêu đúng (FP = merge sai = NGUY HIỂM)
- **Recall**: Trong số nên merge, bao nhiêu được auto-merge

### Cho LLM threshold:
- **Coverage**: Bao nhiêu cặp được gửi đến LLM
- **Efficiency**: Tỷ lệ LLM calls có ý nghĩa (vs noise)

## Domain: Lịch sử Việt Nam

Các sample tập trung vào:
- Các triều đại phong kiến (Lý, Trần, Lê, Nguyễn)
- Các cuộc kháng chiến (Bắc thuộc, Pháp, Mỹ)
- Các nhân vật lịch sử
- Các sự kiện quan trọng
- Văn hóa, xã hội qua các thời kỳ

## Text Format

Mỗi node có text từ 300-800 ký tự, mô phỏng các đoạn văn thực tế từ:
- Sách giáo khoa lịch sử
- Bài viết Wikipedia
- Tài liệu nghiên cứu

## Thresholds được test

### AUTO_MERGE_THRESHOLD:
- 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80

### LLM_THRESHOLD:
- 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
