# Project Plan

Kế hoạch thực hiện đề tài **Phân tích cảm xúc theo khía cạnh và sinh báo cáo tổng hợp từ phản hồi khách hàng bằng Generative AI**, bám theo timeline 6 tuần trong [`docs/Requirements.md`](../docs/Requirements.md) và bù các điểm còn thiếu trong [`docs/Proposal.md`](../docs/Proposal.md).

> Cập nhật trạng thái bằng cách đổi `[ ]` → `[x]` khi hoàn thành từng việc, và cập nhật bảng tổng quan + dòng "Last updated" bên dưới.

**Last updated:** 2026-07-19

## Tổng quan trạng thái

| Tuần | Nội dung | Trạng thái |
|------|----------|------------|
| 1 | Nghiên cứu nền tảng | 🟢 Hoàn thành |
| 2 | Chốt Proposal | 🟡 Đang làm (còn phân công nhóm) |
| 3 | Data pipeline & baseline | 🟢 Hoàn thành |
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
- [x] **Chốt dataset chính**: giữ **SemEval-2014 Laptop** (đã cân nhắc phương án thay thế trong [`docs/Dataset-Options.md`](../docs/Dataset-Options.md)); nhóm đã có tài khoản tải bản gốc từ [trang chính thức](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools).
- [ ] Phân công thành viên nhóm. — **cần nhóm điền tên/vai trò**, xem `docs/Proposal.md` mục "Phân Công Nhóm".
- [x] Cập nhật `docs/Proposal.md` theo các bổ sung trên.

## Tuần 3 — Data pipeline & baseline

- [x] Setup repo/environment (`requirements.txt`, `.venv`, cấu trúc `src/`, `scripts/`, `tests/`).
- [x] Viết data loader cho SemEval-2014 Laptop (`src/data/semeval_loader.py`) + tiền xử lý (`src/data/preprocess.py`), có test (`tests/test_semeval_loader.py`) chạy trên fixture mẫu vì chưa có data thật.
- [x] Xây baseline TF-IDF + Logistic Regression (`src/baseline/tfidf_logreg.py`) cho aspect sentiment classification, CLI chạy qua `scripts/train_baseline.py`.
- [x] Log kết quả baseline thật — chạy trên Kaggle (`notebooks/baseline_semeval_laptop_kaggle_run.ipynb`), dataset `charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis`, dev-split 15% (seed=42) từ `Laptop_Train_v2.xml` (3045 câu, 2358 aspect term → train 2036 / dev 322 example).

  **Kết quả baseline (TF-IDF + Logistic Regression)**: Accuracy **0.6211** | Macro-F1 **0.4266**

  | Polarity | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | positive | 0.74 | 0.72 | 0.73 | 130 |
  | negative | 0.64 | 0.68 | 0.66 | 132 |
  | neutral | 0.32 | 0.31 | 0.32 | 54 |
  | conflict | 0.00 | 0.00 | 0.00 | 6 |

  Nhận xét: model gần như không nhận diện được lớp `conflict` (chỉ 6/2036 example trong train — quá ít để học), và `neutral` cũng yếu (F1 0.32) — dữ liệu lệch lớp nặng (`positive`/`negative` chiếm đa số). Đây là mốc so sánh (baseline) cho model BERT ở Tuần 4 — kỳ vọng BERT cải thiện rõ nhất ở 2 lớp `neutral`/`conflict` nhờ hiểu ngữ cảnh tốt hơn TF-IDF. Chi tiết: `results/baseline_metrics.json`.

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
