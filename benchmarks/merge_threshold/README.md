# Benchmark: EXTRACTION_MERGE_THRESHOLD Evaluation

## Mục đích

Đánh giá và tìm giá trị tối ưu cho `EXTRACTION_MERGE_THRESHOLD` trong `config/extraction.py`.

Threshold này quyết định khi nào một relation mới được LLM đề xuất sẽ:
- **MERGE** với relation có sẵn (similarity >= threshold)
- **CREATE NEW** relation (similarity < threshold)

## Vấn đề hiện tại

- Giá trị hiện tại: `0.85` (cảm tính)
- Không có dữ liệu thực nghiệm chứng minh hiệu quả
- Cần benchmark với các cặp relation thực tế từ domain lịch sử Việt Nam

## Cấu trúc

```
benchmarks/merge_threshold/
├── README.md                    # File này
├── samples.py                   # 100 sample pairs với ground truth
├── test_threshold.py            # Script benchmark
├── results/                     # Kết quả benchmark
│   └── threshold_analysis.md    # Báo cáo phân tích
└── visualizations/              # Biểu đồ (optional)
```

## Cách chạy

```bash
# Chạy benchmark
python benchmarks/merge_threshold/test_threshold.py

# Chạy với verbose
python benchmarks/merge_threshold/test_threshold.py --verbose
```

## Metrics đánh giá

1. **Accuracy**: Tỷ lệ quyết định đúng (merge/separate)
2. **Precision**: Trong số các cặp được merge, bao nhiêu đúng
3. **Recall**: Trong số các cặp nên merge, bao nhiêu được merge
4. **F1 Score**: Harmonic mean của Precision và Recall

## Thresholds được test

- 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
