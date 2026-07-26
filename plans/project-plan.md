# Project Plan

Kế hoạch thực hiện đề tài **Phân tích cảm xúc theo khía cạnh và sinh báo cáo tổng hợp từ phản hồi khách hàng bằng Generative AI**, bám theo timeline 6 tuần trong [`docs/Requirements.md`](../docs/Requirements.md) và bù các điểm còn thiếu trong [`docs/Proposal.md`](../docs/Proposal.md).

> Cập nhật trạng thái bằng cách đổi `[ ]` → `[x]` khi hoàn thành từng việc, và cập nhật bảng tổng quan + dòng "Last updated" bên dưới.

**Last updated:** 2026-07-19

## Tổng quan trạng thái

| Tuần | Nội dung | Trạng thái |
|------|----------|------------|
| 1 | Nghiên cứu nền tảng | 🟢 Hoàn thành |
| 2 | Chốt Proposal | 🟢 Hoàn thành |
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
- [x] Phân công thành viên nhóm (theo `plans/task.txt`) — xem `docs/Proposal.md` mục "Phân Công Nhóm". Còn thiếu người phụ trách Tuần 5 (đánh giá) và Tuần 6 (báo cáo/slides).
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

  **Kết quả DistilBERT (aspect sentiment classification, sequence-pair `[CLS] sentence [SEP] aspect [SEP]`, class-weighted loss, 3 seed 42/43/44 × tối đa 20 epoch + early stopping, lr=3e-5)** — chạy trên Kaggle (`notebooks/finetune_distilbert_semeval_laptop.ipynb`, dataset `dattm03/genai-dataset`, đã bỏ nhãn `conflict` → chỉ còn `positive`/`negative`/`neutral`); kết quả đã chạy: `notebooks-output/finetune_distilbert_semeval_laptop_output.ipynb`.

  Accuracy trung bình **0.7447 ± 0.0159** | Macro-F1 trung bình **0.6861 ± 0.0235** (seed tốt nhất: seed 44, Macro-F1 0.7031)

  | Label | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | negative | 0.747 | 0.808 | 0.775 | 132 |
  | neutral | 0.492 | 0.432 | 0.458 | 54 |
  | positive | 0.841 | 0.810 | 0.825 | 130 |

  Nhận xét: cải thiện rõ so với baseline (Accuracy 0.6211 / Macro-F1 0.4266 — số baseline đo trên bản 4 lớp cũ có `conflict` nên không hoàn toàn tương đồng, cần baseline chạy lại trên data 3 lớp để so sánh chuẩn). `neutral` vẫn là lớp yếu nhất.

  **Kết quả BERT-base** (cùng quy trình, lr=2e-5) — chạy bởi Sơn trên Kaggle (`notebooks/finetune_bert_semeval_laptop.ipynb`); kết quả đã chạy: `notebooks-output/finetune_bert_semeval_laptop_output.ipynb`.

  Accuracy trung bình **0.7627 ± 0.0239** | Macro-F1 trung bình **0.7123 ± 0.0324** (seed tốt nhất: seed 44, Macro-F1 0.7452)

  | Label | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | negative | 0.774 | 0.823 | 0.797 | 132 |
  | neutral | 0.501 | 0.506 | 0.502 | 54 |
  | positive | 0.873 | 0.808 | 0.839 | 130 |

  **So sánh DistilBERT vs BERT-base**: BERT tốt hơn ở cả 3 lớp, rõ nhất ở `neutral` (F1 0.502 vs 0.458) — model lớn hơn hiểu ngữ cảnh tốt hơn ở lớp khó nhất. **Chọn BERT-base (seed 44) làm model chính** cho các bước tiếp theo (tổng hợp thống kê, report generation); DistilBERT giữ lại làm phương án nhẹ/nhanh hơn nếu cần.

  **Còn thiếu**: phần "aspect extraction" — hiện đang dùng aspect term gold có sẵn trong XML, chưa tự extract aspect từ câu thô, sẽ cần cho Tuần 5 khi test trên Amazon Reviews (không có gold aspect).
- [x] Xây bước tổng hợp thống kê theo từng aspect.

  Logic gộp số liệu thuần Python (`src/report/aspect_stats.py`, có test ở `tests/test_aspect_stats.py`, chạy local không cần GPU). Inference + tổng hợp thật chạy trên Kaggle: `notebooks/aspect_stats_semeval_laptop.ipynb`, chạy trên union train+valid+test (2313 example). Đã chạy 2 lần, giữ cả 2 kết quả để so sánh:

  | Model | Majority-sentiment agreement | Notebook đã chạy | File kết quả |
  |---|---|---|---|
  | DistilBERT (seed 44) | 228/253 = 90.12% | `notebooks-output/aspect_stats_semeval_laptop_output.ipynb` | `output/aspect_stats.txt` |
  | **BERT-base (seed 44)** | **236/253 = 93.28%** | `notebooks-output/aspect-level-sentiment-statistics-bert-output.ipynb` | `output/aspect_stats_bert.txt` |

  - 253 aspect có ≥2 lượt nhắc tới (cả 2 lần chạy).
  - BERT cải thiện agreement so với DistilBERT (93.28% vs 90.12%), khớp với việc BERT có Macro-F1 cao hơn ở bước fine-tune, đặc biệt lớp `neutral`. **Dùng `output/aspect_stats_bert.txt` (BERT) làm bảng chính thức bàn giao**, giữ bản DistilBERT lại chỉ để đối chiếu.
  - Top aspect theo số lượt nhắc (ổn định ở cả 2 lần chạy): `screen` (60, positive), `price` (56, positive), `use` (53, positive), `battery life` (52, positive), `keyboard` (50, positive), `battery` (47, **negative**), `warranty` (31, **neutral**), `hard drive`/`windows` (negative).
  - Lưu ý: per-example agreement in trong notebook (0.9421 với BERT, 0.8617 với DistilBERT) đo trên union train+valid+test nên **cao hơn ảo** so với accuracy thật của model (đo trên test set riêng: BERT 0.7627, DistilBERT 0.7447) — không dùng số này để so sánh/báo cáo hiệu năng model, chỉ dùng để sanity-check bước tổng hợp.
  - Bàn giao `output/aspect_stats_bert.txt` (mảng `predicted`) cho bước FLAN-T5 report generation bên dưới.
- [x] **[Pivot bổ sung] Aspect Sentiment Triplet Extraction (ASTE) bằng T5** — theo yêu cầu của teammate, để có "reason" (câu/cụm từ opinion) đi kèm mỗi cặp aspect+sentiment, phục vụ báo cáo tốt hơn. Đây là hướng **bổ sung**, không thay thế mục fine-tune DistilBERT/BERT ở trên (`plans/task.txt` vẫn phân công riêng cho Đạt/Sơn).

  Task: model sinh trực tiếp `aspect: X | opinion: Y | sentiment: Z` từ câu thô — không cần gold aspect term, nên **giải quyết luôn phần "aspect extraction" còn thiếu** ở mục trên. Dataset: SemEval Triplet (`14res`/`15res`/`16res`, domain **nhà hàng** — khác domain Laptop dùng ở trên). 3 model so sánh, cùng hyperparameter (epoch=20, lr=3e-4, effective batch=16) để so sánh công bằng — chỉ đổi checkpoint:

  | Model | Test triplet-F1 | Test P / R | Test Exact match | Notebook đã chạy |
  |---|---|---|---|---|
  | t5-small | 0.7240 | 0.7238 / 0.7243 | 0.6358 | `notebooks-output/train-t5-small-for-aste-on-14res-15res-16res-output.ipynb` |
  | **t5-base** | **0.7442** | 0.7609 / 0.7282 | **0.6481** | `notebooks-output/train-t5-base-for-aste-on-14res-15res-16res-output.ipynb` |
  | flan-t5-base | 0.5898 | 0.5910 / 0.5887 | 0.4877 | `notebooks-output/train-flan-t5-base-for-aste-on-14res-15res-16r-output.ipynb` |

  **t5-base thắng ở mọi metric** — **chốt dùng t5-base làm model chính cho ASTE** ở các bước tiếp theo. **flan-t5-base kém bất ngờ**: log training cho thấy loss ban đầu rất cao (22.1 ở epoch ~1.2, so với t5-small/t5-base ổn định ngay từ đầu) rồi giảm dần suốt 20 epoch nhưng không kịp hồi phục — dấu hiệu điển hình của **lr=3e-4 quá cao cho FLAN-T5** (model đã instruction-tuned thường cần lr thấp hơn nhiều so với T5 gốc khi fine-tune tiếp). Đây là finding đáng ghi vào báo cáo, không phải do FLAN-T5 kém hơn về bản chất.

  Đang thử lại flan-t5-base với `lr=1e-4` (giảm 10 lần, vẫn là giá trị cố định, không thêm warmup/schedule) để kiểm chứng — notebook: `notebooks/train-flan-t5-base-for-aste-on-14res-15res-16res.ipynb`. Không chặn các bước tiếp theo (đã chốt t5-base), chỉ để có kết luận chắc chắn hơn cho báo cáo.

  **Aspect + top-10-reasons bằng t5-base — hoàn tất.** Ban đầu định chạy 100% local (để nối thẳng vào report generation, không qua Kaggle) nhưng t5-base beam-search trên CPU ước tính mất ~3-3.5 tiếng cho 4550 câu → chuyển hẳn inference sang Kaggle GPU (`notebooks/aste_aspect_reasons_restaurant.ipynb`), giữ nguyên logic parse ASTE + tổng hợp thuần Python (`src/data/aste_loader.py`, `src/report/aspect_stats.py::aggregate_aspect_reasons`) có test local, inline vào notebook để chạy GPU — đã verify code inline khớp 100% với bản đã test.

  Kết quả: `output/aspect_reasons_restaurant.json` — 4550 câu (train+dev+test, 3 domain), 619 aspect gold / 585 aspect predicted (≥2 lượt nhắc). **Majority-sentiment agreement: 531/546 = 97.25%** — cao hơn hẳn track BERT/Laptop (90-93%), vì model sinh trực tiếp cả 3 field cùng lúc thay vì phải suy luận sentiment cho aspect đã biết trước.

  Mỗi aspect giờ có đủ count + top-10 reason cho từng sentiment, ví dụ `food` (827 lượt, positive): reasons `great`(109), `good`(100), `delicious`(39)...; negative: `mediocre`(10), `bad`(7), `overpriced`(5)... — sẵn sàng bàn giao `output/aspect_reasons_restaurant.json` (mảng `predicted`) cho bước FLAN-T5 report generation, thay thế/bổ sung cho `output/aspect_stats_bert.txt` (track Laptop, không có reason).

  **Còn thiếu**: kết quả retry flan-t5-base (lr=1e-4) — không chặn gì, chỉ để có kết luận chắc chắn hơn cho báo cáo.
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
