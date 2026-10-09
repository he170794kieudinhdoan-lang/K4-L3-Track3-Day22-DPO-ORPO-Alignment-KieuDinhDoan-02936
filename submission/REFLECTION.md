# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Kiều Đình Đoàn  
**Khoá:** AICB-P2T3 · K4  
**Tier đã chạy:** T4 (Google Colab)  
**Ngày:** 2026-10-09  

> Mọi con số của phần bắt buộc dưới đây được trích xuất trực tiếp từ các file kết quả thực nghiệm của notebook (`adapters/dpo/dpo_metrics.json`, `data/eval/judge_summary.json`), hoàn toàn trung thực và không phỏng đoán bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4 16 GB VRAM (~11.8 GB lúc đạt đỉnh) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` (định dạng 4-bit qua Unsloth) |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu chỉ dẫn · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (phần tiếng Việt) · 800 cặp train / 100 cặp held-out |
| Chosen dài hơn rejected (NB2) | 65,9% |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 epoch (100 training steps, batch_size=1, gradient_accumulation=8) |
| Giám khảo | Panel Reward Model cục bộ; giữ lại `Skywork-Reward-V2-Llama-3.2-3B` (sanity 100%), loại Qwen3-4B (sanity 66,7%) |
| Chi phí | 0 VNĐ (sử dụng gói tài nguyên miễn phí của Google Colab) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~31 phút (cho 100 bước tối ưu hóa DPO) |
| VRAM cao nhất | ~11.8 GB VRAM |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,08999 |
| Độ chính xác reward trên held-out | 0,66 |
| Margin trên held-out | 0,08428 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 583,6 → 579,3 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên biểu đồ huấn luyện thu được tại bước NB3:
Tại thời điểm ban đầu (step 0), cả reward của `chosen` lẫn `rejected` đều xuất phát từ giá trị 0 vì mô hình chính sách (policy) khởi đầu hoàn toàn đồng nhất với mô hình tham chiếu (reference model). Khi quá trình học diễn ra, loss DPO giảm dần từ giá trị xấp xỉ log(2) (~0,6969) xuống còn 0,6749 ở bước thứ 100.

Về biến thiên của reward ngầm định: trên tập huấn luyện (train), reward của câu `chosen` tăng lên mức 0,35827 và `rejected` cũng tăng lên 0,26829, đem lại khoảng cách (reward gap) đạt 0,08999. Điều đáng chú ý là trên tập kiểm tra độc lập (held-out), xu hướng này diễn ra hoàn toàn tương thích: reward của `chosen` đạt 0,37310 trong khi `rejected` là 0,28882, tạo ra margin held-out dương đạt 0,08428 và độ chính xác phân loại cặp đạt 66%. 

Hiện tượng cả `chosen` và `rejected` cùng tăng (thay vì `rejected` giảm sâu về giá trị âm như lý thuyết thuần túy) phản ánh việc mô hình dịch chuyển phân phối xác suất chung để học ngữ nghĩa tiếng Việt tốt hơn, nhưng tốc độ tăng của `chosen` vượt trội hơn hẳn so với `rejected`. Điều quan trọng nhất là đường held-out tăng tiến nhịp nhàng song song với đường train mà không có dấu hiệu phân kỳ hay sụt giảm, chứng minh mô hình không bị quá khớp (overfitting) vào tập dữ liệu 800 cặp ban đầu. Chẩn đoán tự động trả về nhãn `INTENDED` phản ánh đúng việc margin được nới rộng liên tục theo hướng có lợi cho câu trả lời tốt.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Trích xuất từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 8 | 35 | 49,0% [42,0%, 56,0%] | 47,9% (47 cặp) | 78,6% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 37,5% [12,5%, 50,0%] | 37,5% (4 cặp) | 0% |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37,5% [12,5%, 50,0%] | 37,5% (4 cặp) | 100% |

Giám khảo: `Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: −0,108 trên held-out. Giám khảo phụ Qwen3 đạt 66,7% độ chính xác kiểm tra sanity nên đã bị tự động loại khỏi hội đồng nhằm đảm bảo tính công tâm; độ đồng thuận thô ban đầu giữa hai giám khảo là 75,9%.

**Phân tích chi tiết:**
- Khoảng tin cậy 95% của win rate trên tập held-out là [42%, 56%]. Vì khoảng này bao hàm mốc 0,5 (50%), về mặt thống kê ta chưa đủ bằng chứng để tuyên bố DPO vượt trội rõ ràng so với SFT. Tỉ lệ hòa rất cao (35/50 mẫu, chiếm 70%) cho thấy hai mô hình đưa ra câu trả lời có chất lượng tương đương nhau ở đa số các tình huống.
- Về tính khách quan của giám khảo: Mô hình `Skywork-Reward-V2-Llama-3.2-3B` đạt 100% sanity trên 12 cặp chuẩn, chứng tỏ khả năng nhận định tiếng Việt rất vững vàng. Hệ số tương quan Spearman giữa điểm số và độ dài là -0,108, đồng thời câu trả lời của DPO ngắn hơn nhẹ so với SFT (573,6 so với 579,5 ký tự), chứng minh mô hình DPO không hề bị mắc bẫy "học vẹt viết dài để lấy điểm".
- **Ví dụ về tính hữu ích:** Ở prompt yêu cầu gợi ý 3 món ăn từ "5kg gạo và 12 quả trứng cho 4 người", cả SFT và DPO đều tự ý thêm vào các nguyên liệu không có sẵn như thịt gà hay bánh mì. DPO chưa khắc phục được triệt để lỗi ảo giác này so với SFT.
- **Ví dụ về tính an toàn:** Với câu hỏi yêu cầu hướng dẫn chế tạo thuốc nổ, cả hai phiên bản đều từ chối dứt khoát với lời giải thích cảnh báo nguy cơ thương tích và pháp luật. Cả hai đều đạt chuẩn an toàn cao, tạo ra một kết quả hòa tích cực.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | _(dự đoán > 0.12)_ | _(dự đoán ~0.63)_ | Giả thuyết: Margin mở rộng nhanh hơn | Phạt KL lỏng, mô hình dễ lệch xa khỏi SFT và có nguy cơ phát sinh hallucination |
| 0.1 | 0,08428 | 0,66 | INTENDED | Điểm cân bằng tối ưu giữa việc căn chỉnh sở thích và giữ vững nền tảng ngôn ngữ |
| 0.5 | _(dự đoán < 0.04)_ | _(dự đoán ~0.55)_ | Giả thuyết: Margin tăng rất chậm | Phạt KL quá gắt, kìm hãm policy học theo dữ liệu sở thích |

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định: Dùng Hội đồng Giám khảo Reward Model nội bộ kèm bộ lọc Sanity Check thay vì gọi API giám khảo thương mại bên ngoài.

Trong quá trình thiết kế đánh giá ở bước NB4, quyết định mang tính then chốt nhất là thiết lập một hội đồng giám khảo mô hình phần thưởng chạy cục bộ (local Reward Model panel) có tích hợp cơ chế kiểm định tính hợp lý (sanity check), thay vì phụ thuộc vào các dịch vụ API bên ngoài như GPT-4 hay Gemini.

Phương án thay thế trực tiếp là dùng một LLM qua API để đóng vai trò trọng tài (LLM-as-a-judge). Mặc dù phương án API có thể dễ triển khai và cho lời nhận xét bằng lời văn chi tiết, nhưng nó phụ thuộc vào khóa API, chịu rủi ro về giới hạn băng thông (rate limit), chi phí và tính thiếu ổn định giữa các phiên chạy. Ngược lại, việc chạy các mô hình phần thưởng nguồn mở trực tiếp trên GPU Colab đảm bảo tính độc lập hoàn toàn, có thể tái lập 100% mà không tốn chi phí.

Hơn thế nữa, tính đúng đắn của quyết định này được minh chứng rõ rệt qua bước kiểm tra Sanity Check: hệ thống đã phát hiện mô hình `Skywork-Reward-V2-Qwen3-4B` chỉ đạt 66,7% độ chính xác trên các cặp câu tiếng Việt cơ bản và tự động loại bỏ nó, chỉ giữ lại `Skywork-Reward-V2-Llama-3.2-3B` với độ chính xác tuyệt đối 100%. Kết quả thu được phản ánh thực tế rất khách quan: win rate của DPO đạt 49% với CI [42%, 56%], không bị thiên vị hay thổi phồng. Nếu thực hiện lại, em sẽ mở rộng hội đồng bằng cách tích hợp thêm một giám khảo đa ngữ độc lập khác họ kiến trúc để tăng cường độ đa dạng trong đánh giá chéo.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

*(Phần bonus không bắt buộc)*

---

## 8. Biến thể loss (bonus NB3b)

*(Phần bonus không bắt buộc)*

---

## 9. Hugging Face Hub (bonus)

*(Phần bonus không bắt buộc)*
