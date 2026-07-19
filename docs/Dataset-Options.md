# Đề Xuất Dataset (Tham Khảo — Đã Chốt Giữ Nguyên)

> **Quyết định**: nhóm giữ **SemEval-2014 Task 4 (Laptop)** làm dataset chính, tải qua tài khoản chính thức tại [alt.qcri.org/semeval2014/task4 — Data and Tools](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). File này được giữ lại làm tham khảo nếu sau này cần đổi hướng.

Các phương án thay thế/bổ sung từng được cân nhắc:

## So sánh

| Dataset | Ngôn ngữ / Domain | Nhãn | Quy mô | Ưu điểm | Nhược điểm |
|---|---|---|---|---|---|
| **SemEval-2014 Laptop** (đang chọn) | Tiếng Anh, laptop | Aspect term + polarity (4 lớp) | ~3K câu | Chuẩn kinh điển, nhiều repo/paper tham khảo (đã chọn repo nền `BERT-E2E-ABSA`). | Không có aspect category cố định, không có khía cạnh "giao hàng" như ví dụ minh họa ban đầu của nhóm. |
| **SemEval-2014 Restaurant** | Tiếng Anh, nhà hàng | Aspect term + aspect **category** cố định (FOOD, SERVICE, PRICE, AMBIENCE, ANECDOTES) + polarity | Lớn hơn Laptop | Có category cố định → dễ tổng hợp thống kê theo khía cạnh hơn; cùng bộ công cụ/tooling với Laptop. | Vẫn tiếng Anh, domain khác ví dụ minh họa (nhà hàng thay vì sản phẩm điện tử). |
| **MAMS** (Multi-Aspect Multi-Sentiment) | Tiếng Anh, nhà hàng/laptop | Aspect term + polarity, có bản aspect-category (ACSA) | ~13.8K instance | Mỗi câu **luôn có ≥2 aspect với sentiment trái chiều** — đúng insight "khen cái này chê cái kia" nhóm nêu trong proposal; thử thách hơn, dễ tạo điểm nhấn "cải tiến". | Khó hơn để đạt điểm số cao, ít code mẫu có sẵn hơn SemEval. |
| **UIT-ViSFD** ⭐ | **Tiếng Việt**, điện thoại (TMĐT) | 10 aspect category cố định (CAMERA, SCREEN, BATTERY, PERFORMANCE, FEATURES, DESIGN, PRICE, GENERAL, SER&ACC, OTHERS) + polarity, dạng `{ASPECT#polarity}` | 11,122 comment (train 7,786 / dev 1,112 / test 2,224) | Khớp ngôn ngữ + domain với ví dụ minh họa ban đầu (review tiếng Việt, TMĐT); có category cố định dễ tổng hợp thống kê; miễn phí cho mục đích nghiên cứu ([repo GitHub](https://github.com/LuongPhan/UIT-ViSFD), [HuggingFace](https://huggingface.co/datasets/visolex/ViSFD)). | Không có category "SHIPPING/giao hàng" riêng (gần nhất là SER&ACC); phải đổi model tiếng Anh sang tiếng Việt: PhoBERT thay BERT, ViT5/BARTpho thay FLAN-T5 cho report generation — phát sinh thêm việc khảo sát model tiếng Việt. |
| **Amazon Reviews 2023** (đã có trong proposal, vai trò phụ) | Tiếng Anh, đa ngành hàng | Không có nhãn aspect/sentiment sẵn | 571M review (rất lớn) | Quy mô lớn, có nhắc "delivery" tự nhiên trong text. | Không benchmark được, chỉ dùng demo (tự gán nhãn bằng model đã train). |

## Khuyến nghị cá nhân (assistant)

Nếu nhóm muốn bám sát đúng ví dụ minh họa ban đầu (review tiếng Việt, có khía cạnh giao hàng) → chọn **UIT-ViSFD** làm chính. Nếu muốn giữ tooling tiếng Anh có sẵn nhiều hơn và ít rủi ro kỹ thuật hơn → giữ **SemEval-2014 Laptop** hoặc đổi sang **Restaurant** (có category cố định, dễ tổng hợp báo cáo hơn Laptop). **MAMS** là lựa chọn tăng độ khó/điểm nhấn nếu nhóm tự tin.

## Việc cần làm sau khi họp chốt

- [ ] Họp nhóm, chọn 1 dataset chính (và có thể giữ 1 dataset phụ để demo).
- [ ] Nếu chọn UIT-ViSFD: cập nhật lại phần Method trong `docs/Proposal.md` (đổi BERT → PhoBERT, FLAN-T5 → ViT5/BARTpho) và cập nhật `docs/Research-Notes.md`.
- [ ] Cập nhật mục "Dataset — Xác Nhận & Nguồn Tải" và "Repo Nền Kế Thừa" trong `docs/Proposal.md` theo lựa chọn cuối cùng.
- [ ] Cập nhật `plans/project-plan.md`.
