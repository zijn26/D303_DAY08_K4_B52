# RAG Evaluation Results

## Overall Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.778 |
| answer_relevancy | 0.556 |
| context_recall | 0.886 |
| context_precision | 1.000 |

## A/B Comparison

| Config | faithfulness | answer_relevancy | context_recall | context_precision |
|--------|--------|--------|--------|--------|
| hybrid_rerank | 0.828 | 0.540 | 0.886 | 1.000 |
| dense_only | 0.728 | 0.589 | 0.911 | 1.000 |

## Worst Performers

| Question | faithfulness | answer_relevancy | context_recall | context_precision |
|----------|--------|--------|--------|--------|
| Người bán trên Shopee không được đăng bán những loại sản phẩm nào? | 0.000 | 0.000 | 1.000 | 1.000 |
| Shopee hỗ trợ những phương thức thanh toán nào cho đơn hàng trên Sàn? | 0.500 | 0.928 | 0.625 | 1.000 |
| Nếu người mua yêu cầu Trả hàng/Hoàn tiền cho toàn bộ sản phẩm trong đơn hàng và được hoàn đầy đủ giá trị đã thanh toán, phí vận chuyển ban đầu có được hoàn lại không? | 0.500 | 0.740 | 1.000 | 1.000 |

## Recommendations

- answer_relevancy thấp → câu trả lời lạc đề so với câu hỏi; kiểm tra lại reorder_for_llm/format_context có giữ đúng câu hỏi gốc trong prompt không.
- Xem bảng A/B Comparison ở trên để quyết định có nên bật reranking (use_reranking=True) trong production hay không.
