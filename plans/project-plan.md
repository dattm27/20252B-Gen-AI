# Project Plan

Kế hoạch thực hiện đề tài **Phân tích cảm xúc theo khía cạnh và sinh báo cáo tổng hợp từ phản hồi khách hàng bằng Generative AI**, bám theo timeline 6 tuần trong [`docs/Requirements.md`](../docs/Requirements.md) và bù các điểm còn thiếu trong [`docs/Proposal.md`](../docs/Proposal.md).

> Cập nhật trạng thái bằng cách đổi `[ ]` → `[x]` khi hoàn thành từng việc, và cập nhật bảng tổng quan + dòng "Last updated" bên dưới.

**Last updated:** 2026-07-19

## Tổng quan trạng thái

| Tuần | Nội dung | Trạng thái |
|------|----------|------------|
| 1 | Nghiên cứu nền tảng | 🟢 Hoàn thành |
| 2 | Chốt Proposal | 🟡 Đang làm (còn phân công nhóm + họp chốt dataset) |
| 3 | Data pipeline & baseline | 🔴 Chưa bắt đầu |
| 4 | Model cải tiến & report generation | 🔴 Chưa bắt đầu |
| 5 | Đánh giá & phân tích | 🔴 Chưa bắt đầu |
| 6 | Deliverables | 🔴 Chưa bắt đầu |

Chú thích: 🔴 Chưa bắt đầu · 🟡 Đang làm · 🟢 Hoàn thành

## Tuần 1 — Nghiên cứu nền tảng

- [x] Đọc lý thuyết Aspect-Based Sentiment Analysis (ABSA), phân biệt với sentiment analysis thông thường.
- [x] Khảo sát chi tiết dataset SemEval-2014 Task 4 (domain Laptop): cấu trúc data, nhãn aspect và sentiment.
- [x] Khảo sát các nhóm model:
  - [x] Baseline: TF-IDF + Logistic Regression.
  - [x] DistilBERT/BERT cho aspect extraction và aspect sentiment classification.
  - [x] T5/FLAN-T5 cho sinh report từ bảng thống kê aspect-sentiment.
- [x] Tìm và chọn **1 repo/paper nền** cụ thể để kế thừa, xác định rõ điểm nhóm sẽ cải tiến.

Chi tiết: [`docs/Research-Notes.md`](../docs/Research-Notes.md)

## Tuần 2 — Chốt Proposal

- [x] Bổ sung motivation cho từng method (vì sao chọn, so với lựa chọn khác).
- [x] Giải thích input representation / training / prediction cho từng model.
- [x] Định nghĩa metric đánh giá cụ thể:
  - [x] F1/Accuracy cho aspect extraction & sentiment classification.
  - [x] ROUGE/BERTScore (hoặc human eval) cho chất lượng report sinh ra.
  - [x] Tỉ lệ lỗi bịa số liệu bắt được bởi factual checker.
- [x] Xác nhận dataset tải được thật (kèm link tải cụ thể) — *cho lựa chọn SemEval-2014 Laptop hiện tại*.
- [x] Làm rõ Amazon Reviews chỉ dùng để demo quy mô lớn (không có ground-truth aspect), không dùng để benchmark.
- [ ] **Họp nhóm chốt lại dataset chính** — có đề xuất thay thế (UIT-ViSFD, SemEval Restaurant, MAMS), xem [`docs/Dataset-Options.md`](../docs/Dataset-Options.md). Sau khi chốt, cập nhật lại `docs/Proposal.md` + `docs/Research-Notes.md` nếu đổi dataset.
- [ ] Phân công thành viên nhóm. — **cần nhóm điền tên/vai trò**, xem `docs/Proposal.md` mục "Phân Công Nhóm".
- [x] Cập nhật `docs/Proposal.md` theo các bổ sung trên.

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
