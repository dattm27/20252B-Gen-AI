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

## Dataset

Dataset chính có thể dùng **SemEval-2014 Laptop Reviews** vì có sẵn nhãn aspect và sentiment; sau đó có thể dùng thêm **Amazon Reviews** để demo sinh báo cáo trên lượng review lớn hơn.

## Tuần 1: Nhóm Cần Tìm Hiểu

- Aspect-Based Sentiment Analysis là gì, khác gì sentiment analysis bình thường.
- Dataset SemEval-2014, đặc biệt domain Laptop: cấu trúc data, nhãn aspect và sentiment.
- Các model có thể dùng:
  - TF-IDF + Logistic Regression làm baseline.
  - DistilBERT/BERT cho aspect extraction và aspect sentiment classification.
- T5/FLAN-T5 hoặc cách sinh report từ bảng thống kê aspect-sentiment.
- Tìm repo/code nền chạy ABSA trên SemEval để kế thừa, sau đó xác định rõ nhóm sẽ cải tiến gì.
- Đọc 2–3 paper/repo liên quan và chốt proposal gồm: bài toán, input/output, dataset, model baseline, model cải tiến, metric đánh giá.

## Cải Tiến Chính Dự Kiến

- Không chỉ sentiment toàn review mà phân tích theo từng aspect.
- Sinh báo cáo dựa trên thống kê aspect-sentiment thay vì đưa toàn bộ review vào AI.
- Thêm factual checker đơn giản để kiểm tra report có nói đúng aspect/số liệu tổng hợp hay không.
