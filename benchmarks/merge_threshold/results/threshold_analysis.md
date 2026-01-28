# EXTRACTION_MERGE_THRESHOLD Benchmark Report

**Generated:** 2026-01-28 01:52:50
**Total Samples:** 70
**Optimal Threshold:** 0.6

---

## 1. Dataset Summary

| Category | Count | Should Merge | Description |
|----------|-------|--------------|-------------|
| semantic_synonyms | 12 | Yes (12) | Semantic Synonyms |
| directional_inverses | 10 | No (10) | Directional Inverses |
| granularity_variants | 12 | Yes (8) | Granularity Variants |
| related_distinct | 14 | No (14) | Related Distinct |
| domain_overlaps | 12 | Yes (5) | Domain Overlaps |
| cross_family | 10 | No (10) | Cross Family |

**Positive samples (should_merge=True):** 25
**Negative samples (should_merge=False):** 45
**Boundary cases:** 39

---

## 2. Threshold Comparison

| Threshold | Accuracy | Precision | Recall | F1 | F2 | TP | TN | FP | FN |
|-----------|----------|-----------|--------|----|----|----|----|----|----|
| 0.60 ⭐ | 72.9% | 57.1% | 96.0% | 71.6% | 84.5% | 24 | 27 | 18 | 1 |
| 0.65 | 81.4% | 68.8% | 88.0% | 77.2% | 83.3% | 22 | 35 | 10 | 3 |
| 0.70 | 85.7% | 80.0% | 80.0% | 80.0% | 80.0% | 20 | 40 | 5 | 5 |
| 0.72 | 84.3% | 81.8% | 72.0% | 76.6% | 73.8% | 18 | 41 | 4 | 7 |
| 0.75 | 74.3% | 76.9% | 40.0% | 52.6% | 44.2% | 10 | 42 | 3 | 15 |
| 0.78 | 65.7% | 57.1% | 16.0% | 25.0% | 18.7% | 4 | 42 | 3 | 21 |
| 0.80 | 64.3% | 50.0% | 12.0% | 19.4% | 14.2% | 3 | 42 | 3 | 22 |
| 0.82 | 67.1% | 75.0% | 12.0% | 20.7% | 14.4% | 3 | 44 | 1 | 22 |
| 0.85 | 62.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 44 | 1 | 25 |
| 0.88 | 62.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 44 | 1 | 25 |
| 0.90 | 62.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 44 | 1 | 25 |

**Legend:**
- **Precision**: Of all merges predicted, how many were correct? (High = few false merges)
- **Recall**: Of all should-merge pairs, how many did we catch? (High = few missed merges)
- **F2**: Precision-weighted score (penalizes false positives more)

---

## 3. Optimal Threshold Analysis

### Recommended Threshold: **0.6**

### False Positives (18 cases) - CRITICAL

These pairs were incorrectly predicted to MERGE (similarity >= threshold but should NOT merge):
**These are the most dangerous errors - they corrupt the knowledge graph!**

- **ID 22**: `NEXT_SIBLING` vs `PREVIOUS_SIBLING` (sim=0.905, cat=directional_inverses)
  - Reasoning: Inverse relation! NEXT_SIBLING: A→B, PREVIOUS_SIBLING: B→A về thứ tự. Merge sẽ phá vỡ navigation structure....
- **ID 18**: `HAS_SECTION` vs `BELONGS_TO_SECTION` (sim=0.804, cat=directional_inverses)
  - Reasoning: Inverse relation! HAS_SECTION: Parent→Child, BELONGS_TO_SECTION: Child→Parent. Merge sẽ phá vỡ cấu trúc phân cấp....
- **ID 14**: `PRECEDED_BY` vs `FOLLOWED_BY` (sim=0.803, cat=directional_inverses) 🔥
  - Reasoning: CRITICAL: Ngược hướng thời gian! PRECEDED_BY: B→A, FOLLOWED_BY: A→B. Merge sẽ đảo ngược trình tự thời gian....
- **ID 59**: `EXECUTED_BY` vs `KILLED_BY` (sim=0.721, cat=domain_overlaps)
  - Reasoning: EXECUTED_BY: xử tử theo pháp luật, KILLED_BY: giết tổng quát. Sự phân biệt quan trọng cho phân tích lịch sử pháp lý....
- **ID 24**: `LED_TO` vs `INDIRECTLY_LED_TO` (sim=0.701, cat=granularity_variants) 🔥
  - Reasoning: INDIRECTLY_LED_TO nhấn mạnh tính gián tiếp quan trọng cho phân tích lịch sử. Sự phân biệt này có ý nghĩa học thuật, không nên merge....
- **ID 50**: `SIGNED_TREATY_WITH` vs `NEGOTIATED_WITH` (sim=0.688, cat=domain_overlaps) 🔥
  - Reasoning: SIGNED_TREATY_WITH: kết quả ký kết, NEGOTIATED_WITH: quá trình đàm phán. Đàm phán không nhất thiết dẫn đến ký kết....
- **ID 47**: `SUMMARIZES` vs `CONCLUDES` (sim=0.672, cat=related_distinct) 🔥
  - Reasoning: SUMMARIZES: rút gọn nội dung, CONCLUDES: rút ra kết luận/bài học. Khác chức năng (compression vs inference)....
- **ID 54**: `ALLIED_WITH` vs `SUPPORTED_BY` (sim=0.656, cat=domain_overlaps) 🔥
  - Reasoning: ALLIED_WITH: quan hệ ngang hàng, SUPPORTED_BY: B hỗ trợ A (không ngang hàng). Khác về cấu trúc quan hệ....
- **ID 15**: `RESULTED_IN` vs `CONSEQUENCE_OF` (sim=0.651, cat=directional_inverses) 🔥
  - Reasoning: Ngược hướng! RESULTED_IN: A→B (A gây ra B), CONSEQUENCE_OF: A←B (A là hậu quả của B). Mặc dù cùng family causal nhưng hướng khác nhau hoàn toàn....
- **ID 60**: `EXILED_TO` vs `IMPRISONED_AT` (sim=0.650, cat=domain_overlaps)
  - Reasoning: EXILED_TO: bị đày (có thể không bị giam), IMPRISONED_AT: bị giam giữ. Exile và imprisonment là hai hình thức khác nhau....
- **ID 62**: `PART_OF` vs `DURING` (sim=0.641, cat=cross_family) 🔥
  - Reasoning: PART_OF: quan hệ cấu trúc logic, DURING: quan hệ thời gian. Mặc dù có thể cùng đúng nhưng mô tả khía cạnh khác nhau....
- **ID 53**: `SUCCEEDED_DYNASTY` vs `FOLLOWED_BY` (sim=0.639, cat=domain_overlaps) 🔥
  - Reasoning: SUCCEEDED_DYNASTY: kế thừa quyền lực chính thức, FOLLOWED_BY: chỉ thời gian. SUCCEEDED có ý nghĩa chính trị, không chỉ temporal....
- **ID 37**: `CONTRASTS` vs `CONTRADICTS` (sim=0.635, cat=related_distinct) 🔥
  - Reasoning: CONTRASTS: khác biệt nhưng có thể cùng đúng, CONTRADICTS: không thể cùng đúng. Khác mức độ logic quan trọng....
- **ID 55**: `FOUNDED_BY` vs `LED_BY` (sim=0.632, cat=domain_overlaps)
  - Reasoning: FOUNDED_BY: sáng lập ban đầu, LED_BY: lãnh đạo (có thể không phải người sáng lập). Founder ≠ Leader....
- **ID 17**: `ELABORATES` vs `SUMMARIZES` (sim=0.626, cat=directional_inverses)
  - Reasoning: Ngược logic! ELABORATES: A chi tiết hơn B, SUMMARIZES: A ngắn hơn B. Mặc dù cùng hướng A→B nhưng semantic ngược nhau hoàn toàn....
- **ID 19**: `INTRODUCES` vs `CONCLUDES` (sim=0.617, cat=directional_inverses)
  - Reasoning: Ngược vị trí discourse! INTRODUCES: A ở đầu, CONCLUDES: A ở cuối. Semantic hoàn toàn khác nhau....
- **ID 51**: `COLONIZED_BY` vs `RULED_BY` (sim=0.605, cat=domain_overlaps) 🔥
  - Reasoning: COLONIZED_BY: đô hộ thực dân (khai thác, bóc lột), RULED_BY: cai trị tổng quát. COLONIZED_BY có connotation tiêu cực đặc thù....
- **ID 13**: `LED_TO` vs `CAUSED_BY` (sim=0.603, cat=directional_inverses) 🔥
  - Reasoning: CRITICAL: Ngược hướng! LED_TO: A→B, CAUSED_BY: B→A. Merge sẽ đảo ngược quan hệ nhân quả trong knowledge graph....

### False Negatives (1 cases)

These pairs were incorrectly predicted to NOT MERGE (similarity < threshold but SHOULD merge):
(These are less critical - they just create duplicates that can be fixed later)

- **ID 52**: `UPRISING_AGAINST` vs `OPPOSED_BY` (sim=0.556, cat=domain_overlaps) 🔥
  - Reasoning: UPRISING_AGAINST là trường hợp đặc biệt (vũ trang) của OPPOSED_BY. Trong knowledge graph lịch sử, có thể merge....

---

## 4. Similarity Distribution

### Positive Samples (should_merge=True)

- Min: 0.556
- Max: 0.837
- Mean: 0.734
- Median: 0.741
- Std: 0.063

### Negative Samples (should_merge=False)

- Min: 0.424
- Max: 0.905
- Mean: 0.593
- Median: 0.576
- Std: 0.094

### Overlap Analysis

- **Overlap zone:** 0.556 - 0.837
- **Overlap width:** 0.282
- **Positive samples in overlap:** 25
- **Negative samples in overlap:** 29

---

## 5. Recommendations

### Primary Recommendation: `EXTRACTION_MERGE_THRESHOLD = 0.6`

At this threshold:
- **Accuracy:** 72.9%
- **Precision:** 57.1% (of predicted merges, 57% are correct)
- **Recall:** 96.0% (of should-merge pairs, we catch 96%)
- **False Positives:** 18 (merged when shouldn't - DANGEROUS)
- **False Negatives:** 1 (didn't merge when should - recoverable)

🔴 **High false positive risk** (18 cases) - consider raising threshold

### Action Items

1. Update `config/extraction.py`: `EXTRACTION_MERGE_THRESHOLD = 0.6`
2. Monitor false positive rate in production logs
3. Review any false positive cases manually before deployment

---

## 6. All Sample Similarities

| ID | Category | Relation A | Relation B | Similarity | Should Merge | Boundary |
|----|----------|------------|------------|------------|--------------|----------|
| 22 | directional_inv | NEXT_SIBLING | PREVIOUS_SIBLIN | 0.905 | No |  |
| 9 | semantic_synony | CONTEMPORARY_WI | CONCURRENT_WITH | 0.837 | Yes |  |
| 30 | granularity_var | CONTEMPORARY_WI | EXACTLY_CONTEMP | 0.831 | Yes |  |
| 8 | semantic_synony | ELABORATES | EXPANDS_ON | 0.826 | Yes |  |
| 18 | directional_inv | HAS_SECTION | BELONGS_TO_SECT | 0.804 | No |  |
| 14 | directional_inv | PRECEDED_BY | FOLLOWED_BY | 0.803 | No | 🔥 |
| 10 | semantic_synony | PART_OF | BELONGS_TO | 0.781 | Yes |  |
| 26 | granularity_var | OPPOSED_BY | STRONGLY_OPPOSE | 0.780 | Yes | 🔥 |
| 12 | semantic_synony | SUMMARIZES | CONDENSES | 0.773 | Yes |  |
| 5 | semantic_synony | OPPOSED_BY | RESISTED_BY | 0.771 | Yes |  |
| 4 | semantic_synony | FOLLOWED_BY | COMES_BEFORE | 0.762 | Yes |  |
| 3 | semantic_synony | PRECEDED_BY | COMES_AFTER | 0.752 | Yes |  |
| 6 | semantic_synony | LED_BY | COMMANDED_BY | 0.751 | Yes | 🔥 |
| 1 | semantic_synony | LED_TO | RESULTED_IN | 0.749 | Yes | 🔥 |
| 33 | granularity_var | CONTEXTUALIZES | PROVIDES_HISTOR | 0.748 | Yes | 🔥 |
| 58 | domain_overlaps | REFORMED_BY | CHANGED_BY | 0.741 | Yes | 🔥 |
| 57 | domain_overlaps | INFLUENCED_BY | INSPIRED_BY | 0.739 | Yes | 🔥 |
| 2 | semantic_synony | CAUSED_BY | CONSEQUENCE_OF | 0.737 | Yes | 🔥 |
| 11 | semantic_synony | SUPPORTS | PROVIDES_EVIDEN | 0.728 | Yes |  |
| 29 | granularity_var | PARTICIPAGED_IN | ACTIVELY_PARTIC | 0.725 | Yes | 🔥 |
| 7 | semantic_synony | CONTRIBUTED_TO | HELPED_ACHIEVE | 0.725 | Yes |  |
| 59 | domain_overlaps | EXECUTED_BY | KILLED_BY | 0.721 | No |  |
| 32 | granularity_var | IS_A | IS_SUBTYPE_OF | 0.717 | Yes |  |
| 49 | domain_overlaps | MILITARY_VICTOR | DEFEATED | 0.705 | Yes | 🔥 |
| 24 | granularity_var | LED_TO | INDIRECTLY_LED_ | 0.701 | No | 🔥 |
| 56 | domain_overlaps | CONQUERED_BY | ANNEXED_BY | 0.691 | Yes | 🔥 |
| 50 | domain_overlaps | SIGNED_TREATY_W | NEGOTIATED_WITH | 0.688 | No | 🔥 |
| 27 | granularity_var | DURING | THROUGHOUT | 0.687 | Yes | 🔥 |
| 47 | related_distinc | SUMMARIZES | CONCLUDES | 0.672 | No | 🔥 |
| 54 | domain_overlaps | ALLIED_WITH | SUPPORTED_BY | 0.656 | No | 🔥 |
| 15 | directional_inv | RESULTED_IN | CONSEQUENCE_OF | 0.651 | No | 🔥 |
| 60 | domain_overlaps | EXILED_TO | IMPRISONED_AT | 0.650 | No |  |
| 62 | cross_family | PART_OF | DURING | 0.641 | No | 🔥 |
| 53 | domain_overlaps | SUCCEEDED_DYNAS | FOLLOWED_BY | 0.639 | No | 🔥 |
| 34 | granularity_var | SUPERSEDES | COMPLETELY_REPL | 0.636 | Yes |  |
| 37 | related_distinc | CONTRASTS | CONTRADICTS | 0.635 | No | 🔥 |
| 55 | domain_overlaps | FOUNDED_BY | LED_BY | 0.632 | No |  |
| 17 | directional_inv | ELABORATES | SUMMARIZES | 0.626 | No |  |
| 19 | directional_inv | INTRODUCES | CONCLUDES | 0.617 | No |  |
| 23 | granularity_var | LED_TO | DIRECTLY_CAUSED | 0.610 | Yes | 🔥 |
| 51 | domain_overlaps | COLONIZED_BY | RULED_BY | 0.605 | No | 🔥 |
| 13 | directional_inv | LED_TO | CAUSED_BY | 0.603 | No | 🔥 |
| 28 | granularity_var | SIMILAR_TO | DUPLICATE_OF | 0.593 | No |  |
| 31 | granularity_var | REFERS_TO | EXPLICITLY_CITE | 0.581 | No | 🔥 |
| 66 | cross_family | SIMILAR_TO | CONTEMPORARY_WI | 0.581 | No |  |
| 38 | related_distinc | INTRODUCES | CONTEXTUALIZES | 0.579 | No | 🔥 |
| 42 | related_distinc | CONTINUES | ELABORATES | 0.576 | No |  |
| 40 | related_distinc | EXEMPLIFIES | ELABORATES | 0.576 | No | 🔥 |
| 64 | cross_family | CONTEXTUALIZES | CAUSED_BY | 0.573 | No | 🔥 |
| 39 | related_distinc | DEFINES | ELABORATES | 0.569 | No |  |
| 35 | related_distinc | DURING | CONTEMPORARY_WI | 0.567 | No | 🔥 |
| 41 | related_distinc | VISUALIZES | EXEMPLIFIES | 0.567 | No |  |
| 20 | directional_inv | PREREQUISITE_OF | BUILDS_UPON | 0.566 | No | 🔥 |
| 68 | cross_family | IS_A | EXEMPLIFIES | 0.563 | No | 🔥 |
| 52 | domain_overlaps | UPRISING_AGAINS | OPPOSED_BY | 0.556 | Yes | 🔥 |
| 48 | related_distinc | PRECEDED_BY | CAUSED_BY | 0.554 | No | 🔥 |
| 25 | granularity_var | CONTRIBUTED_TO | MAJOR_FACTOR_IN | 0.547 | No | 🔥 |
| 16 | directional_inv | LED_BY | PARTICIPATED_IN | 0.546 | No | 🔥 |
| 46 | related_distinc | REFERS_TO | SIMILAR_TO | 0.543 | No |  |
| 61 | cross_family | FOLLOWED_BY | RESULTED_IN | 0.527 | No | 🔥 |
| 36 | related_distinc | IS_A | PART_OF | 0.525 | No | 🔥 |
| 45 | related_distinc | OPPOSED_BY | CONTRADICTS | 0.511 | No | 🔥 |
| 67 | cross_family | PRECEDED_BY | BUILDS_UPON | 0.506 | No |  |
| 69 | cross_family | REFERS_TO | ELABORATES | 0.504 | No |  |
| 63 | cross_family | CONTRIBUTED_TO | LED_TO | 0.504 | No | 🔥 |
| 43 | related_distinc | BUILDS_UPON | SUPERSEDES | 0.492 | No | 🔥 |
| 70 | cross_family | SUPERSEDES | CONTRADICTS | 0.458 | No |  |
| 44 | related_distinc | LED_BY | CONTRIBUTED_TO | 0.457 | No |  |
| 65 | cross_family | SUPPORTS | CAUSED_BY | 0.445 | No |  |
| 21 | directional_inv | SUPPORTS | CONTRADICTS | 0.424 | No |  |