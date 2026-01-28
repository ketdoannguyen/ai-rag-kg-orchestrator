# Node Similarity Threshold Benchmark Report

**Generated:** 2026-01-28 20:39:10
**Total Samples:** 100
**Optimal AUTO_MERGE_THRESHOLD:** 0.78
**Optimal LLM_THRESHOLD:** 0.35

---

## 1. Executive Summary

### Recommended Configuration

```python
EXTRACTION_AUTO_MERGE_THRESHOLD = 0.78
EXTRACTION_LLM_THRESHOLD = 0.35
```

### Key Metrics at Optimal Thresholds

| Metric | Value |
|--------|-------|
| Auto-merge Precision | 97.3% |
| Auto-merge Recall | 80.0% |
| Auto-merge F1 | 87.8% |
| Auto-merge False Positives | 1 (⚠️ CRITICAL) |
| LLM Zone Coverage | 56.0% |
| Separate Zone Precision | 100.0% |

---

## 2. Dataset Summary

| Category | Count | Should Merge |
|----------|-------|--------------|
| duplicate | 15 | Yes: 15, No: 0 |
| paraphrase | 20 | Yes: 20, No: 0 |
| subset | 10 | Yes: 10, No: 0 |
| overlapping | 15 | Yes: 0, No: 15 |
| related_distinct | 15 | Yes: 0, No: 15 |
| temporal_different | 10 | Yes: 0, No: 10 |
| cause_effect | 10 | Yes: 0, No: 10 |
| unrelated | 5 | Yes: 0, No: 5 |

**Positive samples (should_merge=True):** 45
**Negative samples (should_merge=False):** 55
**Boundary cases:** 45

---

## 3. Similarity Distribution Analysis

### By Ground Truth (should_merge)

| Metric | Should Merge (True) | Should NOT Merge (False) |
|--------|---------------------|--------------------------|
| Count | 45 | 55 |
| Min | 0.697 | 0.107 |
| Max | 0.997 | 0.855 |
| Mean | 0.869 | 0.519 |
| Median | 0.843 | 0.540 |

### Overlap Zone

**Overlap range:** 0.697 - 0.855
**Overlap width:** 0.159

⚠️ There is overlap between positive and negative samples. This means no threshold can perfectly separate them.

---

## 4. Threshold Comparison

### Top 10 Configurations (sorted by F1, then by FP)

| Auto-Merge | LLM | Precision | Recall | F1 | FP | LLM% | Separate% |
|------------|-----|-----------|--------|----|----|------|-----------|
| 0.75 | 0.35 | 95.5% | 93.3% | 94.4% | 2 | 49.0% | 7.0% |
| 0.75 | 0.40 | 95.5% | 93.3% | 94.4% | 2 | 46.0% | 10.0% |
| 0.75 | 0.45 | 95.5% | 93.3% | 94.4% | 2 | 41.0% | 15.0% |
| 0.75 | 0.50 | 95.5% | 93.3% | 94.4% | 2 | 32.0% | 24.0% |
| 0.75 | 0.55 | 95.5% | 93.3% | 94.4% | 2 | 26.0% | 30.0% |
| 0.75 | 0.60 | 95.5% | 93.3% | 94.4% | 2 | 19.0% | 37.0% |
| 0.72 | 0.35 | 89.8% | 97.8% | 93.6% | 5 | 44.0% | 7.0% |
| 0.72 | 0.40 | 89.8% | 97.8% | 93.6% | 5 | 41.0% | 10.0% |
| 0.72 | 0.45 | 89.8% | 97.8% | 93.6% | 5 | 36.0% | 15.0% |
| 0.72 | 0.50 | 89.8% | 97.8% | 93.6% | 5 | 27.0% | 24.0% |

---

## 5. Error Analysis at Optimal Threshold

### Auto-Merge False Positives (1 cases)

These pairs were auto-merged but SHOULD NOT have been merged:

- **ID 50** (sim=0.855, cat=overlapping)
  - Node A: Trật tự thế giới hai cực I-an-ta tồn tại từ 1945 đến 1991. Giai đoạn đầu (1945-đầu 70s) đặc trưng bở...
  - Node B: Trật tự thế giới hai cực I-an-ta tồn tại từ 1945 đến 1991. Giai đoạn sau (đầu 70s-1991) chứng kiến x...
  - Reasoning: Cùng về Trật tự hai cực nhưng mô tả HAI GIAI ĐOẠN KHÁC NHAU. Không nên merge, tạo relation TEMPORAL (FOLLOWED_BY).

### Auto-Merge False Negatives (9 cases)

These pairs should have been merged but were NOT auto-merged:

- **ID 35** (sim=0.777, zone=LLM, cat=paraphrase)
  - This will be sent to LLM zone instead of auto-merge

- **ID 40** (sim=0.776, zone=LLM, cat=subset)
  - This will be sent to LLM zone instead of auto-merge

- **ID 26** (sim=0.761, zone=LLM, cat=paraphrase)
  - This will be sent to LLM zone instead of auto-merge

- **ID 43** (sim=0.755, zone=LLM, cat=subset)
  - This will be sent to LLM zone instead of auto-merge

- **ID 41** (sim=0.754, zone=LLM, cat=subset)
  - This will be sent to LLM zone instead of auto-merge

- **ID 18** (sim=0.753, zone=LLM, cat=paraphrase)
  - This will be sent to LLM zone instead of auto-merge

- **ID 29** (sim=0.734, zone=LLM, cat=paraphrase)
  - This will be sent to LLM zone instead of auto-merge

- **ID 27** (sim=0.733, zone=LLM, cat=paraphrase)
  - This will be sent to LLM zone instead of auto-merge

- **ID 44** (sim=0.697, zone=LLM, cat=subset)
  - This will be sent to LLM zone instead of auto-merge

---

## 6. All Sample Similarities

| ID | Category | Similarity | Should Merge | Predicted Zone |
|----|----------|------------|--------------|----------------|
| 1 | duplicate | 0.997 | Yes | AUTO_MERGE |
| 5 | duplicate | 0.993 | Yes | AUTO_MERGE |
| 9 | duplicate | 0.992 | Yes | AUTO_MERGE |
| 8 | duplicate | 0.991 | Yes | AUTO_MERGE |
| 11 | duplicate | 0.990 | Yes | AUTO_MERGE |
| 12 | duplicate | 0.990 | Yes | AUTO_MERGE |
| 2 | duplicate | 0.990 | Yes | AUTO_MERGE |
| 4 | duplicate | 0.988 | Yes | AUTO_MERGE |
| 6 | duplicate | 0.988 | Yes | AUTO_MERGE |
| 15 | duplicate | 0.987 | Yes | AUTO_MERGE |
| 10 | duplicate | 0.987 | Yes | AUTO_MERGE |
| 14 | duplicate | 0.985 | Yes | AUTO_MERGE |
| 7 | duplicate | 0.974 | Yes | AUTO_MERGE |
| 3 | duplicate | 0.969 | Yes | AUTO_MERGE |
| 13 | duplicate | 0.965 | Yes | AUTO_MERGE |
| 34 | paraphrase | 0.916 | Yes | AUTO_MERGE |
| 17 | paraphrase | 0.899 | Yes | AUTO_MERGE |
| 24 | paraphrase | 0.898 | Yes | AUTO_MERGE |
| 23 | paraphrase | 0.884 | Yes | AUTO_MERGE |
| 31 | paraphrase | 0.881 | Yes | AUTO_MERGE |
| 21 | paraphrase | 0.874 | Yes | AUTO_MERGE |
| 33 | paraphrase | 0.868 | Yes | AUTO_MERGE |
| 50 | overlapping | 0.855 | No | AUTO_MERGE |
| 42 | subset | 0.843 | Yes | AUTO_MERGE |
| 36 | subset | 0.840 | Yes | AUTO_MERGE |
| 28 | paraphrase | 0.835 | Yes | AUTO_MERGE |
| 19 | paraphrase | 0.823 | Yes | AUTO_MERGE |
| 30 | paraphrase | 0.821 | Yes | AUTO_MERGE |
| 16 | paraphrase | 0.820 | Yes | AUTO_MERGE |
| 38 | subset | 0.811 | Yes | AUTO_MERGE |
| 22 | paraphrase | 0.811 | Yes | AUTO_MERGE |
| 20 | paraphrase | 0.807 | Yes | AUTO_MERGE |
| 25 | paraphrase | 0.805 | Yes | AUTO_MERGE |
| 32 | paraphrase | 0.797 | Yes | AUTO_MERGE |
| 37 | subset | 0.797 | Yes | AUTO_MERGE |
| 45 | subset | 0.787 | Yes | AUTO_MERGE |
| 39 | subset | 0.783 | Yes | AUTO_MERGE |
| 35 | paraphrase | 0.777 | Yes | LLM_DECIDE |
| 40 | subset | 0.776 | Yes | LLM_DECIDE |
| 26 | paraphrase | 0.761 | Yes | LLM_DECIDE |
| 55 | overlapping | 0.759 | No | LLM_DECIDE |
| 43 | subset | 0.755 | Yes | LLM_DECIDE |
| 41 | subset | 0.754 | Yes | LLM_DECIDE |
| 18 | paraphrase | 0.753 | Yes | LLM_DECIDE |
| 29 | paraphrase | 0.734 | Yes | LLM_DECIDE |
| 27 | paraphrase | 0.733 | Yes | LLM_DECIDE |
| 49 | overlapping | 0.730 | No | LLM_DECIDE |
| 60 | overlapping | 0.726 | No | LLM_DECIDE |
| 54 | overlapping | 0.720 | No | LLM_DECIDE |
| 73 | related_distinc | 0.716 | No | LLM_DECIDE |
| 59 | overlapping | 0.704 | No | LLM_DECIDE |
| 52 | overlapping | 0.702 | No | LLM_DECIDE |
| 44 | subset | 0.697 | Yes | LLM_DECIDE |
| 58 | overlapping | 0.663 | No | LLM_DECIDE |
| 64 | related_distinc | 0.661 | No | LLM_DECIDE |
| 53 | overlapping | 0.647 | No | LLM_DECIDE |
| 76 | temporal_differ | 0.645 | No | LLM_DECIDE |
| 56 | overlapping | 0.645 | No | LLM_DECIDE |
| 70 | related_distinc | 0.627 | No | LLM_DECIDE |
| 83 | temporal_differ | 0.626 | No | LLM_DECIDE |
| 47 | overlapping | 0.621 | No | LLM_DECIDE |
| 85 | temporal_differ | 0.606 | No | LLM_DECIDE |
| 84 | temporal_differ | 0.602 | No | LLM_DECIDE |
| 93 | cause_effect | 0.597 | No | LLM_DECIDE |
| 77 | temporal_differ | 0.593 | No | LLM_DECIDE |
| 67 | related_distinc | 0.588 | No | LLM_DECIDE |
| 92 | cause_effect | 0.577 | No | LLM_DECIDE |
| 46 | overlapping | 0.576 | No | LLM_DECIDE |
| 51 | overlapping | 0.574 | No | LLM_DECIDE |
| 48 | overlapping | 0.570 | No | LLM_DECIDE |
| 66 | related_distinc | 0.544 | No | LLM_DECIDE |
| 88 | cause_effect | 0.544 | No | LLM_DECIDE |
| 80 | temporal_differ | 0.540 | No | LLM_DECIDE |
| 57 | overlapping | 0.518 | No | LLM_DECIDE |
| 72 | related_distinc | 0.517 | No | LLM_DECIDE |
| 78 | temporal_differ | 0.505 | No | LLM_DECIDE |
| 90 | cause_effect | 0.500 | No | LLM_DECIDE |
| 81 | temporal_differ | 0.485 | No | LLM_DECIDE |
| 63 | related_distinc | 0.479 | No | LLM_DECIDE |
| 69 | related_distinc | 0.478 | No | LLM_DECIDE |
| 79 | temporal_differ | 0.475 | No | LLM_DECIDE |
| 71 | related_distinc | 0.472 | No | LLM_DECIDE |
| 75 | related_distinc | 0.472 | No | LLM_DECIDE |
| 82 | temporal_differ | 0.469 | No | LLM_DECIDE |
| 94 | cause_effect | 0.452 | No | LLM_DECIDE |
| 95 | cause_effect | 0.437 | No | LLM_DECIDE |
| 65 | related_distinc | 0.415 | No | LLM_DECIDE |
| 74 | related_distinc | 0.414 | No | LLM_DECIDE |
| 89 | cause_effect | 0.411 | No | LLM_DECIDE |
| 62 | related_distinc | 0.403 | No | LLM_DECIDE |
| 68 | related_distinc | 0.395 | No | LLM_DECIDE |
| 87 | cause_effect | 0.372 | No | LLM_DECIDE |
| 91 | cause_effect | 0.359 | No | LLM_DECIDE |
| 86 | cause_effect | 0.341 | No | SEPARATE |
| 61 | related_distinc | 0.266 | No | SEPARATE |
| 98 | unrelated | 0.249 | No | SEPARATE |
| 99 | unrelated | 0.208 | No | SEPARATE |
| 96 | unrelated | 0.190 | No | SEPARATE |
| 100 | unrelated | 0.176 | No | SEPARATE |
| 97 | unrelated | 0.107 | No | SEPARATE |

---

## 7. Recommendations

### Primary Recommendation

```python
# config/extraction.py
EXTRACTION_AUTO_MERGE_THRESHOLD = 0.78
EXTRACTION_LLM_THRESHOLD = 0.35
```

⚠️ **1 false positives** - consider raising AUTO_MERGE_THRESHOLD

### Decision Flow at Optimal Thresholds

```
Similarity >= 0.78  →  AUTO-MERGE (no LLM needed)
Similarity >= 0.35  →  Send to LLM for decision
Similarity < 0.35   →  Keep SEPARATE
```
