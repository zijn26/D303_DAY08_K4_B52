# RAG Evaluation Results

*Đánh giá bằng RAGAS trên 15 câu hỏi (golden_dataset.json) về chính sách thương mại điện tử/hỗ trợ khách hàng (trả hàng-hoàn tiền, thanh toán, vận chuyển, quyền riêng tư), lấy từ `data/landing/legal` và `data/landing/news`. LLM giám khảo + LLM sinh câu trả lời đều qua OpenRouter.*

## Overall Scores

| Metric | Mean | Std | Min | Max | Đánh giá |
|---|---|---|---|---|---|
| Faithfulness | 0.750 | 0.278 | 0.000 | 1.000 | chấp nhận được |
| Answer Relevancy | 0.508 | 0.434 | 0.000 | 0.958 | yếu, cần cải thiện |
| Context Recall | 0.886 | 0.275 | 0.000 | 1.000 | rất tốt |
| Context Precision | 1.000 | 0.000 | 1.000 | 1.000 | rất tốt |

**Phân tích từng metric:**

- **Faithfulness** (câu trả lời có bám đúng nội dung trong context được cấp không (không bịa đặt/suy diễn ngoài nguồn)) — trung bình **0.750**, chấp nhận được. 6/15 câu hỏi dưới ngưỡng 0.7 (độ lệch chuẩn 0.278 cho thấy kết quả dao động mạnh giữa các câu hỏi). Thấp nhất: "Người bán trên Shopee không được đăng bán những loại sản phẩm nào?" (0.000).
- **Answer Relevancy** (câu trả lời có thực sự trả lời đúng trọng tâm câu hỏi đặt ra không) — trung bình **0.508**, yếu, cần cải thiện. 6/15 câu hỏi dưới ngưỡng 0.7 (độ lệch chuẩn 0.434 cho thấy kết quả dao động mạnh giữa các câu hỏi). Thấp nhất: "Nếu thanh toán đơn hàng bằng thẻ tín dụng/ghi nợ, khi hoàn tiền thì ti…" (0.000).
- **Context Recall** (retriever có lấy về đủ thông tin (evidence) cần thiết để trả lời đúng không) — trung bình **0.886**, rất tốt. 3/15 câu hỏi dưới ngưỡng 0.7 (độ lệch chuẩn 0.275 cho thấy kết quả dao động mạnh giữa các câu hỏi). Thấp nhất: "Khi gặp lỗi thanh toán (M01/D01) vì tài khoản Shopee ghi nhận dấu hiệu…" (0.000).
- **Context Precision** (trong số các đoạn context lấy về, bao nhiêu phần trăm thực sự liên quan/hữu ích (không rác)) — trung bình **1.000**, rất tốt. Không có câu hỏi nào dưới ngưỡng 0.7.

## A/B Comparison

So sánh `hybrid_rerank` (semantic + lexical + RRF + rerank) vs `dense_only` (chỉ semantic search, không rerank) — xem `RealRAGPipeline.set_config` trong `eval_pipeline.py` và `retrieve()` trong `src/task9_retrieval_pipeline.py`.

| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|--------|--------|--------|--------|--------|
| hybrid_rerank | 0.750 | 0.508 | 0.886 | 1.000 |
| dense_only | 0.898 | 0.591 | 0.886 | 1.000 |

**Phân tích chênh lệch (hybrid_rerank − dense_only):**

| Metric | Delta | Nhận xét |
|---|---|---|
| Faithfulness | -0.148 | dense_only tốt hơn |
| Answer Relevancy | -0.082 | dense_only tốt hơn |
| Context Recall | +0.000 | gần như không khác biệt |
| Context Precision | +0.000 | gần như không khác biệt |

`dense_only` thắng ở nhiều metric hơn hoặc bằng — reranking hiện KHÔNG cải thiện chất lượng đầu ra tương xứng với chi phí (thêm 1 bước gọi model/API), cân nhắc tắt reranking hoặc đổi RERANK_METHOD/lại trọng số RRF. Lưu ý: golden dataset chỉ có 15 câu hỏi, chênh lệch nhỏ (<0.05) nhiều khả năng là nhiễu thống kê chứ chưa đủ mạnh để kết luận chắc chắn — nên mở rộng golden dataset trước khi đưa ra quyết định cuối cùng về reranking.

## Worst Performers

Xếp hạng theo điểm trung bình 4 metric (thấp nhất trước), kèm trích đoạn answer/ground_truth thực tế để chẩn đoán nguyên nhân.

### "Khi gặp lỗi thanh toán (M01/D01) vì tài khoản Shopee ghi nhận dấu hiệu bất thường, người mua nên xử…" (avg=0.417)

| Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|
| 0.667 | 0.000 | 0.000 | 1.000 |

- **Answer thực tế:** Khi gặp lỗi thanh toán (M01/D01) do tài khoản Shopee ghi nhận dấu hiệu bất thường, người mua nên tham khảo và tuân thủ các Điều Khoản của Shopee. Nếu vấn đề vẫn tiếp diễn, người mua có thể thử sử dụng thiết bị khác để đă…
- **Ground truth:** Gỡ cài đặt các công cụ/phần mềm bất thường hoặc khôi phục lại cài đặt gốc trên thiết bị đang dùng Shopee, sau đó thực hiện thanh toán lại như giao dịch bình thường, đồng thời tham khảo Điều khoản dịch vụ của Shopee.
- **Context #1 lấy về:** / **Lỗi (M01/D01): Tài khoản Shopee của bạn ghi nhận dấu hiệu bất thường
- **Chẩn đoán:** điểm yếu nhất là *Answer Relevancy* (0.000) — khả năng do câu trả lời lạc đề hoặc quá chung chung so với câu hỏi.

### "Người bán trên Shopee không được đăng bán những loại sản phẩm nào?" (avg=0.500)

| Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|
| 0.000 | 0.000 | 1.000 | 1.000 |

- **Answer thực tế:** Tôi không thể xác minh thông tin này từ nguồn hiện có.
- **Ground truth:** Hàng giả, hàng nhái, hàng nhập lậu, hàng vi phạm quyền sở hữu trí tuệ, và các hàng hóa/dịch vụ/thông tin vi phạm pháp luật hoặc quy định của Shopee.
- **Context #1 lấy về:** đồng mua bán giữa họ, việc đăng bán hàng hóa, bảo hành sản phẩm và tương tự. Shopee không can thiệp vào giao dịch giữa các Người Sử Dụng. Shopee có thể hoặc không sàng lọc trước Người Sử Dụng hoặc Nội Dung hoặc thông tin…
- **Chẩn đoán:** điểm yếu nhất là *Faithfulness* (0.000) — khả năng do generation (task10) trả lời không bám sát context được cấp.

### "Nếu người mua chưa nhận được hàng nhưng hệ thống đã cập nhật trạng thái 'Giao hàng thành công', ngườ…" (avg=0.667)

| Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|
| 0.667 | 0.000 | 1.000 | 1.000 |

- **Answer thực tế:** Nếu người mua chưa nhận được hàng nhưng hệ thống đã cập nhật trạng thái 'Giao hàng thành công', người mua nên liên hệ với Shopee để được hỗ trợ. Hiện tại, Shopee không hỗ trợ giao hoặc lấy hàng theo giờ cụ thể hoặc sớm h…
- **Ground truth:** Nếu vẫn trong 24 giờ kể từ lúc trạng thái bị cập nhật sai, tiếp tục chờ đơn vị vận chuyển giao hàng; nếu đã quá 24 giờ mà vẫn chưa nhận được hàng, bấm Trả hàng/Hoàn tiền và chọn lý do 'Chưa nhận được hàng' để Shopee xử l…
- **Context #1 lấy về:** **3. Bạn chưa nhận được hàng nhưng hệ thống lại cập nhật đã giao hàng thành công:**
- **Chẩn đoán:** điểm yếu nhất là *Answer Relevancy* (0.000) — khả năng do câu trả lời lạc đề hoặc quá chung chung so với câu hỏi.

### "Mã giảm giá do Người bán phát hành (Shop Voucher) có được hoàn lại khi yêu cầu Trả hàng/Hoàn tiền đư…" (avg=0.687)

| Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|
| 0.750 | 0.000 | 1.000 | 1.000 |

- **Answer thực tế:** Mã giảm giá do Người bán phát hành (Shop Voucher) sẽ không được hoàn lại trong bất kỳ trường hợp nào, ngay cả khi yêu cầu Trả hàng/Hoàn tiền được chấp nhận. Tuy nhiên, bạn có thể liên hệ với Người bán để được hỗ trợ về S…
- **Ground truth:** Không, Shop Voucher (mã do Người bán phát hành) và Mã miễn phí vận chuyển sẽ không được hoàn lại trong bất cứ trường hợp nào; người mua có thể chủ động liên hệ Người bán để được hỗ trợ.
- **Context #1 lấy về:** Trả hàng/hoàn tiền) nếu Voucher đã hết hiệu lực hoặc hết lượt sử dụng. Shop Voucher (mã do Người bán phát hành) và Mã miễn phí vận chuyển sẽ không được hoàn lại trong bất cứ trường hợp nào. Tuy nhiên, bạn có thể chủ động…
- **Chẩn đoán:** điểm yếu nhất là *Answer Relevancy* (0.000) — khả năng do câu trả lời lạc đề hoặc quá chung chung so với câu hỏi.

### "Nếu thanh toán đơn hàng bằng thẻ tín dụng/ghi nợ, khi hoàn tiền thì tiền sẽ về đâu và mất bao lâu?" (avg=0.750)

| Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|
| 1.000 | 0.000 | 1.000 | 1.000 |

- **Answer thực tế:** Nếu bạn thanh toán đơn hàng bằng thẻ tín dụng/ghi nợ, tiền hoàn sẽ được gửi về đúng tài khoản tín dụng/ghi nợ mà bạn đã sử dụng để thanh toán đơn hàng. Thời gian nhận tiền hoàn sẽ mất từ 7 đến 14 ngày làm việc, tùy thuộc…
- **Ground truth:** Tiền được hoàn về đúng thẻ tín dụng/ghi nợ đã dùng để thanh toán, thời gian nhận được là 7-14 ngày làm việc tùy theo ngân hàng.
- **Context #1 lấy về:** * Đối với đơn hàng thanh toán bằng Thẻ Tín dụng/Ghi nợ, Shopee chỉ hỗ trợ hoàn tiền về đúng tài khoản Tín dụng/Ghi nợ đã sử dụng khi thanh toán đơn hàng.   * Đối với hình thức hoàn tiền về Tài khoản Ngân hàng, bạn cần đả…
- **Chẩn đoán:** điểm yếu nhất là *Answer Relevancy* (0.000) — khả năng do câu trả lời lạc đề hoặc quá chung chung so với câu hỏi.

## Recommendations

- **Generation (task10):** answer_relevancy trung bình chỉ 0.508 — nhiều câu trả lời lạc đề/quá dài dòng so với câu hỏi gốc; kiểm tra `format_context`/`reorder_for_llm` có giữ đúng câu hỏi ở vị trí LLM chú ý nhất không.
- Xem mục *A/B Comparison* để quyết định có nên bật reranking (`use_reranking=True`) trong production hay không — hiện tại chênh lệch giữa 2 config khá nhỏ.
- Golden dataset hiện có 15 câu hỏi — nên mở rộng thêm (đặc biệt các câu hỏi thuộc nhóm điểm thấp ở mục Worst Performers) để kết luận A/B đáng tin cậy hơn.
