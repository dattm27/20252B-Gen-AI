# Báo Cáo Kiểm Tra Nguồn Dataset (SemEval-2014 Task 4 — Laptop)

Ghi lại quá trình kiểm tra 3 nguồn dataset đã lấy được, để chọn ra nguồn dùng chính thức cho pipeline. Tất cả kiểm tra được thực hiện bằng cách đọc trực tiếp file (không suy đoán) — parse XML/JSONL, so khớp nội dung câu, đối chiếu offset.

## Tóm tắt kết quả

| # | Nguồn | Có gold test? | Vấn đề | Kết luận |
|---|---|---|---|---|
| 1 | **Chính thức** — [alt.qcri.org/semeval2014/task4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) (tài khoản riêng) | ❌ (2 file test tải được đều là bản "blind") | Train đầy đủ, đúng chuẩn; nhưng thiếu gold polarity cho test | ✅ **Dùng làm train chính thức** |
| 2 | **Kaggle** — `charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis` | ❌ | Là **bản mirror y hệt nguồn 1** (byte-identical), chỉ convert thêm sang CSV | Không thêm giá trị cho vấn đề gold test; giữ lại phần Restaurant domain làm dữ liệu dự phòng |
| 3 | **HuggingFace** — `NEUDM/semeval-2014` (laptop) | ✅ (nhưng...) | Test split (118 câu) thực chất **116/118 trùng với TRAIN chính thức**, không phải test gold thật; chỉ dùng 1482/3033 câu train chính thức (~49%), thiếu offset ký tự, license không rõ | ❌ Không dùng — tự tách dev-split từ nguồn 1 tốt hơn (nhiều dữ liệu hơn, offset chuẩn, nguồn chính thức) |

## Chi tiết kiểm tra

### 1. Nguồn chính thức (alt.qcri.org)

File tải về: `Laptop_Train_v2.xml`, `Laptops_Train.xml` (bản cũ, thừa), `Laptops_Test_Data_PhaseA.xml`, `Laptops_Test_Data_phaseB.xml`, `SemEval14_ABSA_AnnotationGuidelines.pdf`.

- `Laptop_Train_v2.xml`: **3045 câu** (3033 câu unique text), có đầy đủ `aspectTerm term/polarity/from/to`. Đây là bản train chuẩn, dùng làm nguồn chính.
- `Laptops_Test_Data_PhaseA.xml`: 800 câu, **0 aspectTerm** — bản blind cho subtask 1 (aspect extraction).
- `Laptops_Test_Data_phaseB.xml`: 800 câu, có 654 aspect term nhưng **`polarity` = None cho toàn bộ** — bản blind cho subtask 2 (polarity).
- → Cả 2 file test đều là bản dùng để **nộp bài thi**, không phải đáp án (gold). Không tìm thấy mục "Gold" riêng trên trang đã tải.

### 2. Kaggle (`charitarth/...`)

File tải về (đặt tạm ở `data/kaggle`, sau đó đã gộp phần dùng được vào `data/raw`): các file `.xml` + bản `.csv` tương ứng cho cả Laptop và Restaurant.

- Diff byte-by-byte: `Laptop_Train_v2.xml` và `Laptops_Test_Data_phaseB.xml` trên Kaggle **giống hệt 100%** file từ nguồn 1.
- `Laptops_Test_Data_PhaseA.csv` / `PhaseB.csv`: chỉ có cột `id, Sentence` — xác nhận vẫn là bản blind, không có `Aspect Term`/`polarity`.
- Giá trị duy nhất Kaggle mang lại: có sẵn domain **Restaurant** (`Restaurants_Train_v2.xml`, `Restaurants_Test_Data_PhaseA/B.xml`) mà nguồn 1 chưa tải — giữ lại ở `data/raw/restaurant_bonus/` để dự phòng, không dùng ngay.

### 3. HuggingFace (`NEUDM/semeval-2014`)

Tải trực tiếp `laptop/{train,dev,test}.jsonl` (1060/310/118 dòng). Format: `input` (câu), `output` (chuỗi kiểu `"[['aspect','polarity']]"`), không có offset ký tự.

Kiểm tra bằng cách so khớp nội dung câu với 2 tập chính thức (train 3033 câu, test 800 câu):

| NEUDM split | Trùng TRAIN chính thức | Trùng TEST chính thức | Không trùng cái nào |
|---|---|---|---|
| train (1056 unique) | 1031 | 1 | 25 |
| dev (310 unique) | 305 | 0 | 5 |
| test (118 unique) | **116** | **0** | 2 |

→ **Test split của NEUDM gần như hoàn toàn lấy từ TRAIN chính thức**, không phải bộ test gold thật của SemEval. Về bản chất, NEUDM chỉ tự chia lại (train/dev/test) một phần dữ liệu train chính thức — nhưng chỉ dùng 1482/3033 câu (~49%), bỏ phí gần nửa dữ liệu có sẵn, không có offset, và là nguồn re-format không chính thức (tác giả cá nhân, license không ghi rõ). Không có lý do để dùng thay vì tự tách dev-split từ nguồn 1 (nhiều dữ liệu hơn, offset chuẩn, nguồn chính thức).

## Dataset được chọn (mô tả chính thức cho Proposal / Báo cáo)

**SemEval-2014 Task 4 — Laptop domain**, tải từ nguồn chính thức [alt.qcri.org/semeval2014/task4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools), lưu tại `data/raw/train/Laptop_Train_v2.xml`.

- **Quy mô**: 3045 câu review laptop tiếng Anh (3033 câu unique), annotate thủ công.
- **Nhãn**: mỗi câu có 0..n `aspectTerm`, mỗi aspect term gồm `term` (cụm từ), `polarity` ∈ {positive, negative, neutral, conflict}, `from`/`to` (offset ký tự trong câu).
- **Vì chưa có bộ test gold chính thức**: pipeline tự tách **dev-split ngẫu nhiên (mặc định 15%, `random_state=42`)** từ chính 3045 câu train để đánh giá baseline/model — đã cài sẵn trong `scripts/train_baseline.py` (tham số `--dev-ratio`, `--seed`). Số liệu vì vậy chỉ dùng để **so sánh nội bộ** giữa baseline và các model cải tiến của nhóm, không so sánh trực tiếp được với các con số published trên leaderboard SemEval (vốn dùng đúng 800 câu test gold chính thức).
- **Dữ liệu dự phòng**: domain Restaurant (từ Kaggle, `data/raw/restaurant_bonus/`) và Amazon Reviews 2023 (xem `docs/Proposal.md`) nếu cần mở rộng phạm vi demo sau này.
- **Việc còn để ngỏ (không chặn tiến độ)**: nếu sau này tìm được bộ gold test chính thức (800 câu `Laptops_Test_Gold.xml`), chỉ cần truyền `--test data/raw/test/Laptops_Test_Gold.xml` vào script để dùng ngay, không cần sửa code.

## File đã dọn / không dùng

- `data/raw/legacy/Laptops_Train.xml` — bản train cũ (3048 câu, lệch so với v2), không dùng.
- `data/raw/neudm/` — dữ liệu đã tải để kiểm tra, **không đưa vào pipeline** theo kết luận ở trên.
- `data/raw/restaurant_bonus/` — domain Restaurant dự phòng từ Kaggle, không dùng vì nhóm đã chốt chỉ dùng domain Laptop.

**Cập nhật 2026-07-21**: đã xoá cả 3 thư mục trên khỏi `data/raw/` (dọn dẹp local, không ảnh hưởng git vì `data/raw/*` đã gitignore từ đầu). Dataset Laptop đang dùng được mirror lại tại [kaggle.com/datasets/dattm03/genai-dataset](https://www.kaggle.com/datasets/dattm03/genai-dataset).
