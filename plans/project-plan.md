# Project Plan

Kế hoạch thực hiện đề tài **Phân tích cảm xúc theo khía cạnh và sinh báo cáo tổng hợp từ phản hồi khách hàng bằng Generative AI**, bám theo timeline 6 tuần trong [`docs/Requirements.md`](../docs/Requirements.md) và bù các điểm còn thiếu trong [`docs/Proposal.md`](../docs/Proposal.md).

> Cập nhật trạng thái bằng cách đổi `[ ]` → `[x]` khi hoàn thành từng việc, và cập nhật bảng tổng quan + dòng "Last updated" bên dưới.

**Last updated:** 2026-07-19

## Tổng quan trạng thái

| Tuần | Nội dung | Trạng thái |
|------|----------|------------|
| 1 | Nghiên cứu nền tảng | 🔴 Chưa bắt đầu |
| 2 | Chốt Proposal | 🔴 Chưa bắt đầu |
| 3 | Data pipeline & baseline | 🔴 Chưa bắt đầu |
| 4 | Model cải tiến & report generation | 🔴 Chưa bắt đầu |
| 5 | Đánh giá & phân tích | 🔴 Chưa bắt đầu |
| 6 | Deliverables | 🔴 Chưa bắt đầu |

Chú thích: 🔴 Chưa bắt đầu · 🟡 Đang làm · 🟢 Hoàn thành

## Tuần 1 — Nghiên cứu nền tảng

- [ ] Đọc lý thuyết Aspect-Based Sentiment Analysis (ABSA), phân biệt với sentiment analysis thông thường.
- [ ] Khảo sát chi tiết dataset SemEval-2014 Task 4 (domain Laptop): cấu trúc data, nhãn aspect và sentiment.
- [ ] Khảo sát các nhóm model:
  - [ ] Baseline: TF-IDF + Logistic Regression.
  - [ ] DistilBERT/BERT cho aspect extraction và aspect sentiment classification.
  - [ ] T5/FLAN-T5 cho sinh report từ bảng thống kê aspect-sentiment.
- [ ] Tìm và chọn **1 repo/paper nền** cụ thể để kế thừa, xác định rõ điểm nhóm sẽ cải tiến.

## Tuần 2 — Chốt Proposal

- [ ] Bổ sung motivation cho từng method (vì sao chọn, so với lựa chọn khác).
- [ ] Giải thích input representation / training / prediction cho từng model.
- [ ] Định nghĩa metric đánh giá cụ thể:
  - [ ] F1/Accuracy cho aspect extraction & sentiment classification.
  - [ ] ROUGE/BERTScore (hoặc human eval) cho chất lượng report sinh ra.
  - [ ] Tỉ lệ lỗi bịa số liệu bắt được bởi factual checker.
- [ ] Xác nhận dataset tải được thật (kèm link tải cụ thể).
- [ ] Làm rõ Amazon Reviews chỉ dùng để demo quy mô lớn (không có ground-truth aspect), không dùng để benchmark.
- [ ] Phân công thành viên nhóm.
- [ ] Cập nhật `docs/Proposal.md` theo các bổ sung trên.

## Tuần 3 — Data pipeline & baseline

- [ ] Setup repo/environment.
- [ ] Viết data loader cho SemEval-2014 Laptop, tiền xử lý dữ liệu.
- [ ] Xây baseline TF-IDF + Logistic Regression cho aspect sentiment classification.
- [ ] Log kết quả baseline để làm mốc so sánh.

## Tuần 4 — Model cải tiến & report generation

- [ ] Fine-tune DistilBERT/BERT cho aspect extraction + sentiment classification.
- [ ] Xây bước tổng hợp thống kê theo từng aspect.
- [ ] Dùng FLAN-T5 sinh báo cáo ngắn từ bảng thống kê.
- [ ] Xây factual checker đơn giản đối chiếu số liệu trong report với thống kê gốc.

## Tuần 5 — Đánh giá & phân tích

- [ ] So sánh baseline vs model cải tiến theo các metric đã chốt.
- [ ] Test report generation trên Amazon Reviews để demo quy mô lớn.
- [ ] Đánh giá hiệu quả của factual checker.
- [ ] Error analysis, rút ra insight/khám phá mới.

## Tuần 6 — Deliverables

- [ ] Đưa code lên GitHub, viết `README.txt` hướng dẫn cài đặt và chạy chương trình.
- [ ] Viết báo cáo LaTeX (Overleaf) theo cấu trúc yêu cầu: giới thiệu chủ đề, bài toán, chi tiết ứng dụng model/phương pháp, kết quả đánh giá.
- [ ] Làm slides trình bày.
