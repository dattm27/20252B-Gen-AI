# Chủ đề

**Phân tích cảm xúc theo khía cạnh và sinh báo cáo tổng hợp từ phản hồi khách hàng bằng Generative AI**

Hiểu đơn giản: thay vì chỉ phân loại một review là tích cực/tiêu cực, hệ thống sẽ xác định khách đang khen/chê cụ thể điểm nào.

## Ví dụ

Review:

> "Máy có màn hình đẹp, giao hàng nhanh nhưng pin yếu."

Hệ thống sẽ phân tích:

| Khía cạnh   | Sentiment |
|-------------|-----------|
| Màn hình    | Tích cực  |
| Giao hàng   | Tích cực  |
| Pin         | Tiêu cực  |

Sau đó, khi có nhiều review, hệ thống tổng hợp và sinh báo cáo như:

> "Khách hàng đánh giá tích cực về màn hình và tốc độ giao hàng. Tuy nhiên, pin là khía cạnh có nhiều phản hồi tiêu cực nhất và cần được ưu tiên cải thiện."

## Dự kiến Pipeline

1. Input là các review khách hàng.
2. Trích xuất các khía cạnh được nhắc tới: pin, màn hình, giá, giao hàng, ...
3. Phân loại sentiment cho từng khía cạnh.
4. Tổng hợp thống kê theo từng khía cạnh.
5. Dùng mô hình sinh văn bản để tạo báo cáo ngắn, kèm kiểm tra để tránh AI bịa số liệu/insight.

## Bài Toán (Input / Output)

- **Input**: một tập review khách hàng (văn bản tiếng Anh, câu hoặc đoạn ngắn) về sản phẩm (laptop).
- **Output**:
  1. Danh sách khía cạnh (aspect) được nhắc tới trong từng review, kèm sentiment (positive/negative/neutral/conflict) cho từng khía cạnh.
  2. Một báo cáo tổng hợp dạng văn bản ngắn, tóm tắt xu hướng khen/chê theo từng khía cạnh trên toàn bộ tập review.

## Phương Pháp (Method)

### 1. Baseline — TF-IDF + Logistic Regression

- **Motivation**: mốc so sánh đơn giản, không cần GPU, giúp định lượng mức độ cải thiện khi dùng model ngữ cảnh (BERT).
- **Input representation**: câu review → vector TF-IDF (n-gram 1-2) trên cửa sổ từ quanh aspect term đã biết.
- **Training**: Logistic Regression đa lớp (positive/negative/neutral/conflict), scikit-learn.
- **Prediction**: input 1 vector TF-IDF → xác suất từng lớp sentiment.
- **Hạn chế**: không tự trích xuất aspect term, không hiểu ngữ cảnh xa (negation, mệnh đề đối lập).

### 2. Aspect Extraction + Sentiment Classification — Fine-tuned BERT/DistilBERT

- **Motivation**: cần một model vừa tìm được aspect term trong câu (không có sẵn) vừa phân loại sentiment cho từng aspect, trong khi vẫn hiểu ngữ cảnh toàn câu (ví dụ "pin yếu nhưng màn hình đẹp").
- **Input representation**: câu review → BERT tokenizer (WordPiece) → embedding theo token.
- **Training**: fine-tune trên SemEval-2014 Laptop với unified tagging scheme (BIEOS) — mỗi token được gán nhãn kết hợp vị trí-trong-aspect và polarity (ví dụ `B-POS`, `I-NEG`), theo cách tiếp cận của repo nền (mục "Repo Nền").
- **Prediction**: câu mới → chuỗi nhãn theo token → decode ra danh sách (aspect term, polarity).

### 3. Report Generation — FLAN-T5

- **Motivation**: FLAN-T5 là model text-to-text đã được instruction-tune, cho kết quả tốt trên các tác vụ table-to-text/data-to-text mà không cần fine-tune nhiều, phù hợp để chuyển bảng thống kê → câu văn tự nhiên.
- **Input representation**: bảng thống kê aspect-sentiment (ví dụ: "screen: 80% positive (12 reviews); battery: 65% negative (9 reviews)") được serialize thành chuỗi văn bản có cấu trúc, kèm prompt hướng dẫn.
- **Training**: dùng ở chế độ few-shot/prompt-based trước; nếu còn thời gian sẽ fine-tune nhẹ trên một tập report mẫu do nhóm tự viết.
- **Prediction**: input là bảng thống kê (dạng text) → output là đoạn báo cáo ngắn 2-4 câu.

### 4. Factual Checker

- **Motivation**: model sinh văn bản có thể "bịa" số liệu không khớp thống kê gốc — cần một lớp kiểm tra trước khi xuất báo cáo.
- **Cách làm**: rule-based — dùng regex trích các con số (%, số lượng) và tên aspect xuất hiện trong report, đối chiếu với bảng thống kê gốc; nếu lệch hoặc nhắc tới aspect không có trong thống kê → gắn cờ cảnh báo.

## Kết Quả Baseline (Tuần 3)

Đã train baseline TF-IDF + Logistic Regression thật trên `Laptop_Train_v2.xml` (3045 câu, dev-split 15% giữ lại làm test tạm — xem `docs/Dataset-Verification-Report.md`):

| Polarity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.74 | 0.72 | 0.73 | 130 |
| negative | 0.64 | 0.68 | 0.66 | 132 |
| neutral | 0.32 | 0.31 | 0.32 | 54 |
| conflict | 0.00 | 0.00 | 0.00 | 6 |

**Accuracy: 0.6211 | Macro-F1: 0.4266**. Baseline gần như không học được lớp `conflict` (chỉ 6 example trong tập train) và yếu ở `neutral` — dữ liệu lệch lớp nặng, TF-IDF không nắm được ngữ cảnh phủ định/tương phản trong câu. Đây là mốc so sánh cho model BERT ở Tuần 4. Notebook: `notebooks/baseline_semeval_laptop_kaggle_run.ipynb`, số liệu đầy đủ: `results/baseline_metrics.json`.

## Metric Đánh Giá

- **Aspect extraction**: Precision / Recall / F1 theo span (exact match, dùng `seqeval`).
- **Aspect sentiment classification**: Accuracy + Macro-F1 (do phân bố nhãn lệch).
- **Report generation**: ROUGE-L / BERTScore so với report mẫu do nhóm tự viết, kèm đánh giá thủ công (thang 1–5: mạch lạc, hữu ích).
- **Factual checker**: % số liệu/aspect trong report khớp với bảng thống kê gốc.

## Dataset — Xác Nhận & Nguồn Tải

- **Chính: SemEval-2014 Task 4 (Laptop)** — có sẵn nhãn aspect term + polarity. Tải trực tiếp từ trang chính thức [alt.qcri.org/semeval2014/task4 — Data and Tools](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) (nhóm đã có tài khoản để tải bản gốc). Nếu cần bản dự phòng, dữ liệu XML cũng được mirror công khai trong repo nền [lixin4ever/BERT-E2E-ABSA](https://github.com/lixin4ever/BERT-E2E-ABSA) (thư mục `data/laptop14`).
- **Phụ: Amazon Reviews 2023** (McAuley Lab) — [huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), tải qua `datasets.load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", trust_remote_code=True)`. **Lưu ý**: dataset này không có nhãn aspect/sentiment sẵn → chỉ dùng để demo chạy pipeline ở quy mô lớn (tự động gán nhãn bằng model đã train), **không** dùng để tính metric benchmark.

Chi tiết khảo sát đầy đủ (lý thuyết ABSA, so sánh các repo, nguồn tham khảo) xem tại [`Research-Notes.md`](Research-Notes.md).

## Repo Nền Kế Thừa

**[lixin4ever/BERT-E2E-ABSA](https://github.com/lixin4ever/BERT-E2E-ABSA)** (Apache-2.0) — model end-to-end ABSA (BERT + CRF/self-attention, unified tagging BIEOS), có sẵn dataset `laptop14`. Được chọn thay vì `songyouwei/ABSA-PyTorch` (chỉ phân loại sentiment khi đã biết trước aspect, không tự trích xuất) hoặc `ScalaConsultants/Aspect-Based-Sentiment-Analysis` (thư viện đóng gói sẵn, khó chỉnh sửa sâu).

## Cải Tiến Chính Dự Kiến (So Với Repo Nền)

1. Không chỉ dừng ở 1 câu — thêm bước **tổng hợp thống kê** aspect-sentiment trên nhiều review.
2. Thêm **module sinh report** (FLAN-T5) từ bảng thống kê thay vì đưa toàn bộ review vào AI — repo nền không có bước này.
3. Thêm **factual checker** đơn giản để kiểm tra report có nói đúng aspect/số liệu tổng hợp hay không.

## Phân Công Nhóm

_TODO: điền tên thành viên (4-5 người) và vai trò (data/baseline, model BERT, report generation + factual checker, đánh giá/báo cáo, slides)._
