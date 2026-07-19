# Research Notes — Tuần 1

Ghi chú tìm hiểu nền tảng cho đề tài ABSA + sinh báo cáo. Nguồn tham khảo được liệt kê ở cuối mỗi mục.

## 1. Aspect-Based Sentiment Analysis (ABSA) là gì

Sentiment analysis thông thường gán **một nhãn cảm xúc cho cả câu/văn bản** (positive/negative/neutral). ABSA chi tiết hơn: với một câu, nó xác định **từng khía cạnh (aspect)** được nhắc tới và **cảm xúc riêng cho khía cạnh đó**. ABSA thường được chia thành các subtask:

1. **Aspect Term Extraction (ATE)** — tìm cụm từ chỉ khía cạnh trong câu (ví dụ "battery life", "build quality").
2. **Aspect Term Polarity** — phân loại sentiment (positive/negative/neutral/conflict) cho từng aspect term đã tìm được.
3. **Aspect Category Detection** — xác định aspect thuộc category cố định nào (ví dụ domain restaurant: FOOD, SERVICE, PRICE, AMBIENCE).
4. **Aspect Category Polarity** — sentiment cho từng category.

Domain **Laptop** trong SemEval-2014 chỉ có nhãn cho subtask 1 và 2 (aspect term + polarity), không có aspect category — phù hợp với hướng "trích xuất khía cạnh tự do" mà nhóm định làm.

Nguồn: [SemEval-2014 Task 4 – Task Description](https://alt.qcri.org/semeval2014/task4/)

## 2. Dataset: SemEval-2014 Task 4 (Laptop)

- Hơn 3K câu tiếng Anh trích từ customer reviews laptop, annotate thủ công.
- Định dạng **XML**: mỗi `<sentence>` chứa `<aspectTerms>` gồm các `<aspectTerm term="..." polarity="..." from=".." to=".."/>` (vị trí ký tự bắt đầu/kết thúc của aspect trong câu).
- Nhãn polarity có 4 giá trị: `positive`, `negative`, `neutral`, `conflict` (conflict = vừa khen vừa chê cùng 1 khía cạnh trong câu).
- Có bộ train và test riêng (test dùng để chấm điểm cuối, không train trên đó).

**Cách tải**: Trang chính thức (alt.qcri.org) không cho link tải trực tiếp công khai (cần đăng ký nhẹ với ban tổ chức). Tuy nhiên dữ liệu XML gốc được mirror công khai trong nhiều repo học thuật, ví dụ trong `data/laptop14` của repo nền đã chọn (mục 4) — nhóm sẽ dùng bản mirror này để đảm bảo tải được ngay, không phụ thuộc đăng ký.

Nguồn: [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/), [lixin4ever/BERT-E2E-ABSA](https://github.com/lixin4ever/BERT-E2E-ABSA)

## 3. Khảo sát các phương pháp

| Method | Vai trò trong pipeline | Ghi chú |
|---|---|---|
| TF-IDF + Logistic Regression | Baseline | Đơn giản, nhanh, không hiểu ngữ cảnh/vị trí aspect — dùng làm mốc so sánh. |
| BERT/DistilBERT (fine-tuned, tagging scheme BIEOS/CRF) | Aspect extraction + sentiment classification | Model chính, học joint 2 subtask cùng lúc thay vì pipeline 2 bước tách rời. |
| FLAN-T5 | Sinh report từ bảng thống kê aspect-sentiment | Instruction-tuned, mạnh ở text-to-text/table-to-text (đã SOTA trên benchmark ToTTo), phù hợp sinh câu tóm tắt từ dữ liệu có cấu trúc mà không cần fine-tune nhiều. |

Nguồn: [FLAN-T5 tutorial – DataCamp](https://www.datacamp.com/tutorial/flan-t5-tutorial), [Text-to-Text Pre-Training for Data-to-Text Tasks (T5 trên ToTTo)](https://aclanthology.org/2020.inlg-1.14.pdf)

## 4. Repo nền được chọn để kế thừa

**[lixin4ever/BERT-E2E-ABSA](https://github.com/lixin4ever/BERT-E2E-ABSA)** (Apache-2.0)

- Làm **end-to-end ABSA**: một model BERT duy nhất vừa trích xuất aspect term vừa phân loại sentiment, dùng unified tagging scheme (BIEOS/BIO/OT) thay vì 2 model tách rời — khớp với bước 2+3 trong pipeline dự kiến của nhóm.
- Có sẵn dataset `laptop14` (SemEval-2014 Laptop) trong repo.
- Kiến trúc: BERT-Base-Uncased + linear/GRU/self-attention + CRF.
- Có `fast_run.py` để train/replicate kết quả nhanh.

**Lý do chọn thay vì các lựa chọn khác đã khảo sát:**
- `songyouwei/ABSA-PyTorch` (đã archived, MIT) chỉ làm sentiment classification khi đã biết trước aspect term — không giải quyết bước trích xuất aspect, phải ghép thêm 1 model riêng.
- `ScalaConsultants/Aspect-Based-Sentiment-Analysis` là thư viện đóng gói sẵn (khó chỉnh sửa sâu để chứng minh "cải tiến của nhóm").

## 5. Điểm cải tiến dự kiến (so với repo nền)

1. Thêm bước **tổng hợp thống kê** theo aspect trên nhiều review (repo nền chỉ dừng ở mức 1 câu).
2. Thêm **module sinh report** (FLAN-T5) từ bảng thống kê — repo nền không có.
3. Thêm **factual checker**: rule-based, trích số liệu/tên aspect trong report bằng regex rồi đối chiếu với bảng thống kê gốc, cảnh báo nếu sai lệch.

## 6. Metric đánh giá dự kiến

- **Aspect extraction**: Precision / Recall / F1 theo span (exact match), dùng `seqeval`.
- **Aspect sentiment classification**: Accuracy + Macro-F1 (vì nhãn lệch — positive/negative nhiều hơn conflict).
- **Report generation**: ROUGE-L hoặc BERTScore so với vài report mẫu do nhóm tự viết + đánh giá thủ công (thang 1–5: mạch lạc, hữu ích).
- **Factual checker**: % số liệu/aspect trong report khớp với thống kê gốc (đo trên tập report do nhóm sinh ra).

## 7. Dataset phụ: Amazon Reviews

**[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** trên Hugging Face, tải qua `datasets.load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", trust_remote_code=True)`.

⚠️ Dataset này **không có nhãn aspect/sentiment sẵn** → chỉ dùng để **demo** chạy pipeline đã train trên SemEval ở quy mô lớn (tự động gán nhãn rồi sinh report), **không dùng để tính metric benchmark**.

## 8. Hướng factual-checking tham khảo thêm (ngoài scope hiện tại)

Có các hướng phức tạp hơn trong literature (FactCC, ConFactCheck — dùng model riêng để kiểm tra tính nhất quán) nhưng vượt quá phạm vi môn học; nhóm chọn cách rule-based đơn giản (mục 6) là đủ để chứng minh ý tưởng.

Nguồn: [Consistency Is the Key (arXiv 2511.12236)](https://arxiv.org/abs/2511.12236), [FactCC – hallucination detection framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC12796222/)
