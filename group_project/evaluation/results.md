# RAG Evaluation Results

## Overall Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.778 |
| answer_relevancy | 0.556 |
| context_recall | 0.886 |
| context_precision | 1.000 |

## Analysis
- **Context Precision** (1.000 - Hoàn hảo): Các đoạn dữ liệu (chunks) trích xuất về đều liên quan trực tiếp đến câu hỏi và được sắp xếp ở ngay các vị trí đầu tiên, không bị lẫn thông tin nhiễu.
- **Context Recall** (0.886 - Tốt): Bước tìm kiếm (Retrieval) đã thu thập được gần như đầy đủ các thông tin cần thiết từ CSDL để trả lời câu hỏi.
- **Faithfulness** (0.778 - Khá): Mức độ trung thực của câu trả lời so với ngữ cảnh trích xuất được. Khoảng 22% nội dung câu trả lời có nguy cơ bị "bịa" (hallucination) hoặc lấy từ tri thức bên ngoài thay vì bám sát context.
- **Answer Relevancy** (0.556 - Thấp): Độ liên quan giữa câu trả lời và câu hỏi của người dùng. Mức điểm này cho thấy câu trả lời còn lan man, dông dài hoặc chưa đi đúng trọng tâm thắc mắc.

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
