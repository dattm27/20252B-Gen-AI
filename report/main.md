% Phân tích cảm xúc theo khía cạnh và sinh báo cáo tổng hợp từ phản hồi khách hàng bằng Generative AI
% Nhóm 20252B — Đạt, Sơn, Hoàng, Vinh, Hưng

> **Ghi chú**: bản này là bản Markdown/Word để cả nhóm dễ xem và fill tiếp (không cần Overleaf). Các đoạn đánh dấu **[TODO: ...]** là phần do nhóm Hoàng/Vinh/Hưng phụ trách hoặc chưa làm xong. Toàn bộ số liệu lấy trực tiếp từ `plans/project-plan.md` và các notebook đã chạy trong `notebooks-output/`, không có số liệu bịa.
>
> **Phạm vi báo cáo**: báo cáo này tập trung vào một pipeline duy nhất — Aspect Sentiment Triplet Extraction (ASTE) bằng T5, domain nhà hàng. Track phân loại sentiment theo khía cạnh đã biết trước (TF-IDF+LogReg/DistilBERT/BERT, domain laptop) vẫn tồn tại trong code/notebook (`notebooks/finetune_{distilbert,bert}_semeval_laptop.ipynb`, kết quả trong `plans/project-plan.md`) như một bước khảo sát ban đầu của nhóm, nhưng không đưa vào báo cáo nộp để giữ báo cáo tập trung vào một pipeline hoàn chỉnh, nhất quán.

## Tóm tắt

Thay vì chỉ phân loại một review là tích cực/tiêu cực, hệ thống trong báo cáo này xác định cụ thể khách hàng đang khen/chê điểm nào (khía cạnh — *aspect*) **trực tiếp từ câu thô, không cần biết trước khía cạnh**, kèm luôn "lý do" (*opinion*) cho mỗi nhận định, rồi tổng hợp thống kê trên nhiều review và dùng mô hình sinh văn bản để tạo báo cáo tóm tắt ngắn, kèm một bộ kiểm tra tính xác thực (factual checker) để hạn chế mô hình bịa số liệu.

Nhóm sinh trực tiếp bộ ba (khía cạnh, ý kiến, sentiment) từ câu thô — bài toán **Aspect Sentiment Triplet Extraction (ASTE)** — trên domain nhà hàng (SemEval Restaurant Triplet data, 14res+15res+16res), so sánh một baseline tra cứu tần suất tự xây với ba checkpoint T5-family (T5-small, T5-base, FLAN-T5-base) cùng hyperparameter. T5-base cải thiện gần gấp đôi triplet-F1 so với baseline (0.3736 → 0.7442) và được chọn làm model chính. Nhóm cũng ghi lại một số phát hiện đáng chú ý trong quá trình thực nghiệm, trong đó có một kết quả phủ định giả thuyết ban đầu về nguyên nhân FLAN-T5-base hoạt động kém trên tác vụ trích xuất.

---

## 1. Giới thiệu

### 1.1. Chủ đề và động lực

Các hệ thống phân tích cảm xúc truyền thống thường chỉ đưa ra một nhãn tích cực/tiêu cực duy nhất cho toàn bộ review, trong khi một review thực tế thường đề cập nhiều khía cạnh khác nhau với sentiment khác nhau. Với người quản lý nhà hàng, một con số tổng quát ("70% review tích cực") không đủ để biết *nên cải thiện điểm gì* — họ cần biết cụ thể khía cạnh nào đang được khen, khía cạnh nào đang bị chê, và **vì sao**. Đây chính là động lực của bài toán **Aspect-Based Sentiment Analysis** (ABSA), minh hoạ qua ví dụ:

> *"The food was great but the service was slow."*

| Khía cạnh | Sentiment | Lý do (opinion) |
|---|---|---|
| food | Tích cực | great |
| service | Tiêu cực | slow |

Khi có hàng trăm, hàng nghìn review như vậy, việc đọc thủ công từng câu là không khả thi. Nhóm hướng tới một hệ thống tự động hoá toàn bộ quy trình: từ review thô, tự tổng hợp và sinh ra một đoạn báo cáo dạng:

> *"Khách hàng đánh giá tích cực về food (đa số nhắc 'great', 'delicious') và price. Tuy nhiên, service là khía cạnh có nhiều phản hồi tiêu cực nhất (thường bị chê 'slow') và cần được ưu tiên cải thiện."*

Một rủi ro lớn khi dùng mô hình sinh văn bản (generative model) cho bước cuối là hiện tượng *hallucination* — mô hình có thể bịa ra số liệu hoặc nhận định không khớp với dữ liệu gốc. Vì vậy pipeline của nhóm có thêm một bước kiểm tra tính xác thực (factual checker) trước khi xuất báo cáo cuối cùng.

### 1.2. Pipeline tổng quan

![Pipeline tổng quan. Một model duy nhất (T5-base fine-tune cho ASTE) đọc câu thô và sinh trực tiếp cả 3 thông tin (aspect, opinion, sentiment) — không cần bước trích xuất aspect riêng biệt. Khối nét đứt (factual checker) là phần do nhóm Hoàng/Vinh/Hưng phụ trách.](figures/pipeline.png)

Điểm khác biệt cốt lõi so với cách tiếp cận ABSA truyền thống (phân loại sentiment khi đã biết trước aspect): ở đây **một model duy nhất** vừa tìm aspect, vừa tìm opinion, vừa phân loại sentiment, trong cùng một lần sinh văn bản — không cần ghép nối nhiều model con.

### 1.3. Tổng quan các phương pháp

- **Baseline tra cứu tần suất (AsteLookupBaseline)** — mốc so sánh phi-neural cho bài toán trích xuất bộ ba.
- **T5-small / T5-base / FLAN-T5-base fine-tuned cho ASTE** — sinh trực tiếp bộ ba (khía cạnh, ý kiến, sentiment) từ câu thô.
- **FLAN-T5 sinh báo cáo** (đã fine-tune) từ bảng thống kê + reason khía cạnh (Mục 4.4).
- **Factual checker** đối chiếu số liệu và reason trong báo cáo sinh ra với bảng thống kê gốc (Mục 4.5).

---

## 2. Công trình liên quan

### 2.1. Aspect-Based Sentiment Analysis

ABSA thường được chia thành các bài toán con: *Aspect Term Extraction* (tìm cụm từ chỉ khía cạnh trong câu), *Aspect Term Polarity* (phân loại sentiment cho từng aspect term đã tìm được), và ở mức chi tiết hơn — *opinion term extraction* (tìm cụm từ diễn đạt lý do) [1]. Aspect Sentiment Triplet Extraction (ASTE) là bài toán hợp nhất cả ba: sinh ra bộ ba (aspect, opinion, sentiment) trong một bước duy nhất, thay vì pipeline nhiều bước tách rời.

### 2.2. Repo nền: Generative-ABSA

Nhóm chọn **[IsakZhang/Generative-ABSA](https://github.com/IsakZhang/Generative-ABSA)** — code cho bài báo **"Towards Generative Aspect-Based Sentiment Analysis"** (Zhang, Li, Deng, Bing, Lam — **ACL 2021**) [2] — làm repo nền để kế thừa ý tưởng. Bài báo này đóng khung 4 bài toán con của ABSA (AOPE, UABSA, **ASTE**, TASD) dưới dạng sinh văn bản (generative), dùng **T5-base**, đúng domain nhà hàng (Rest14/15/16 — chính là 14res/15res/16res nhóm dùng). Đáng chú ý: hyperparameter mặc định của repo này (`lr=3e-4`, `batch_size=16`, `20 epoch`) trùng khớp với config nhóm đã chọn độc lập cho các notebook fine-tune T5 — một tín hiệu cho thấy lựa chọn hyperparameter của nhóm phù hợp với chuẩn đã được kiểm chứng trong literature.

**Điểm khác của nhóm so với repo nền**: Generative-ABSA dừng lại ở việc sinh bộ ba cho từng câu riêng lẻ (đúng phạm vi một bài báo NLP). Nhóm mở rộng thêm: (1) bước tổng hợp thống kê + xếp hạng top-10 lý do phổ biến nhất theo tần suất trên *nhiều* review; (2) một baseline tra cứu tần suất tự xây (không có trong bài báo gốc) làm mốc so sánh phi-neural; (3) module sinh report bằng FLAN-T5 từ bảng thống kê; (4) một factual checker để kiểm tra report có bịa số liệu hay không; (5) một demo quy mô lớn trên dữ liệu Yelp thực tế, không có gold label.

### 2.3. Dataset: SemEval Restaurant Triplet data

Nhóm dùng bộ dữ liệu SemEval Restaurant Triplet data [3, 4], cụ thể phiên bản `ASTE-Data-V1-AAAI2020`, với cách gán nhãn theo nhóm token (mỗi aspect gán một mã nhóm, opinion tương ứng gán mã nhóm cùng độ dài dạng `S...S`) để một câu có thể chứa nhiều bộ ba độc lập.

### 2.4. Mô hình sinh văn bản: T5 và FLAN-T5

**T5** [5] (Text-to-Text Transfer Transformer) đóng khung mọi tác vụ NLP dưới dạng text-to-text, phù hợp tự nhiên với bài toán sinh bộ ba có cấu trúc từ câu thô. **FLAN-T5** [6] là phiên bản T5 đã được instruction-tuned trên nhiều tác vụ đa dạng, đã cho kết quả tốt trên các benchmark table-to-text/data-to-text như ToTTo [7] mà không cần fine-tune sâu — đây là lý do nhóm chọn FLAN-T5 (thay vì T5 gốc) cho bước sinh báo cáo cuối, dù kết quả thực nghiệm lại cho thấy FLAN-T5-base *không* phải lựa chọn tốt hơn T5-base cho bước trích xuất bộ ba (Mục 6) — một phát hiện thú vị được thảo luận riêng.

### 2.5. Kiểm tra tính xác thực (factual checking)

Trong literature có nhiều hướng phức tạp để kiểm tra tính nhất quán của văn bản sinh ra so với nguồn dữ liệu gốc, ví dụ FactCC [8] (train một classifier riêng để phát hiện câu không nhất quán với văn bản nguồn) hay các phương pháp dùng model riêng để đối chiếu theo từng claim (ConFactCheck và các hướng liên quan [9]). Các hướng này cần huấn luyện thêm một model kiểm tra riêng, vượt quá phạm vi và thời gian của môn học. Nhóm chọn cách tiếp cận rule-based nhẹ hơn nhưng vẫn xử lý được văn xuôi tự do (không ép một mẫu câu cố định): trích mọi con số xuất hiện trong report bằng regex, gán mỗi con số về đúng aspect được nhắc gần nó nhất, rồi đối chiếu trực tiếp với bảng thống kê gốc — chi tiết cụ thể ở Mục 4.5.

---

## 3. Bài toán

- **Input**: một câu review thô (tiếng Anh), *không* có thông tin khía cạnh đi kèm.
- **Output**: danh sách bộ ba (*aspect*, *opinion*, *sentiment*), ví dụ:

  ```
  aspect: price | opinion: reasonable | sentiment: positive
  ```

  Trường *opinion* chính là cụm từ diễn đạt lý do (reason) cho sentiment đó, phục vụ trực tiếp cho bước sinh báo cáo.
- **Dataset**: SemEval Restaurant Triplet data (14res + 15res + 16res).

**Bước tổng hợp và sinh báo cáo** (sau khi có bộ ba trên toàn bộ tập review):

- **Input**: danh sách (aspect, opinion, sentiment) trên toàn bộ tập review.
- **Output**: (1) bảng thống kê theo từng aspect (số lượt positive/negative/neutral, top-10 lý do phổ biến nhất mỗi sentiment), (2) một đoạn báo cáo văn bản ngắn tóm tắt xu hướng, (3) kết quả kiểm tra tính xác thực của báo cáo đó.

---

## 4. Phương pháp

### 4.1. Baseline tra cứu tần suất — AsteLookupBaseline

ASTE không có một baseline TF-IDF/Logistic-Regression tương đương (bài toán cần tự tìm aspect, không phải phân loại khi đã biết trước) — nhóm xây một baseline phi-neural để làm mốc so sánh: không cần train gradient, chỉ ghi nhớ (a) tập các cụm aspect và (b) sentiment phổ biến nhất của từng cụm opinion, cả hai đều thống kê trực tiếp từ tập train. Ở câu test, thuật toán tra cứu nguyên văn các cụm đã thấy trong câu (khớp n-gram dài nhất trước, tối đa 3 từ, quét trái sang phải không chồng lấn), rồi ghép mỗi aspect tìm được với opinion *gần nhất* trên câu (theo khoảng cách token).

Baseline này có recall thấp theo thiết kế: bất kỳ cách diễn đạt aspect hay opinion nào khác với những gì đã thấy trong tập train đều bị bỏ sót hoàn toàn — đây chính là "sàn" để đo mức độ cải thiện thực sự của một model biết tổng quát hoá. Cài đặt: `src/baseline/aste_lookup_baseline.py`, chạy hoàn toàn local (không cần GPU) qua `scripts/eval_aste_baseline.py`.

### 4.2. T5-small / T5-base / FLAN-T5-base fine-tuned cho ASTE

Nhóm sinh trực tiếp bộ ba (aspect, opinion, sentiment) bằng một mô hình text-to-text: câu thô được thêm một prefix hướng dẫn cố định (`"extract aspect sentiment triplets: "`) làm input, target text có định dạng `"aspect: X | opinion: Y | sentiment: Z ; ..."` (nối nhiều bộ ba bằng dấu `;` nếu câu có nhiều aspect).

Ba checkpoint T5-family được fine-tune với *cùng* hyperparameter (20 epoch, lr=3e-4, effective batch size 16 — riêng T5-base dùng batch nhỏ hơn kèm gradient accumulation lớn hơn để giữ effective batch không đổi do cần nhiều bộ nhớ GPU hơn) để so sánh công bằng, chỉ đổi checkpoint: `t5-small`, `t5-base`, `google/flan-t5-base`. Ở phần dự đoán, chuỗi văn bản model sinh ra được parse bằng regex thành danh sách bộ ba có cấu trúc.

Metric đánh giá là *triplet-F1*: Precision/Recall/F1 theo phép giao tập hợp (set overlap) giữa bộ ba dự đoán và bộ ba gốc (đã chuẩn hoá chữ thường), micro-average trên toàn bộ tập test (không average theo từng câu rồi lấy trung bình, để câu có nhiều bộ ba đóng góp đúng tỉ trọng). Hạ tầng: ba notebook Kaggle độc lập, tự động tải dataset (clone GitHub nếu chưa có sẵn trên Kaggle input) — `notebooks/train-{t5-small,t5-base,flan-t5-base}-for-aste-*.ipynb`.

### 4.3. Tổng hợp thống kê theo khía cạnh + top-10 lý do

Bước tổng hợp gộp các bộ ba (aspect, opinion, sentiment) theo aspect: đếm số lượt mỗi sentiment và lấy sentiment đa số; trong mỗi nhóm (aspect, sentiment), các cụm *opinion* được xếp hạng theo tần suất, giữ lại top-10 làm ví dụ "lý do" minh hoạ cho sentiment đó — ví dụ aspect `food` với sentiment `positive` có thể có lý do phổ biến nhất là `great`, `good`, `delicious`.

Toàn bộ logic gộp số liệu này là thuần Python (không cần GPU), được viết và test local (`src/report/aspect_stats.py`, kiểm thử ở `tests/test_aspect_stats.py`); chỉ riêng phần suy luận mô hình (cần GPU) chạy trên Kaggle, sau đó logic tổng hợp đã test được inline nguyên văn vào notebook để đảm bảo đúng kết quả như bản đã kiểm thử local.

### 4.4. Sinh báo cáo bằng FLAN-T5

**Chọn aspect đưa vào báo cáo.** Từ bảng tổng hợp (Mục 4.3), hàm `select_report_rows` chọn ra 4 aspect đại diện để đưa vào một báo cáo ngắn (đưa hết vài trăm aspect vào một đoạn văn là vô nghĩa): (1) aspect được nhắc nhiều nhất, (2) aspect có tỉ lệ positive/total cao nhất trong số 30 aspect được nhắc nhiều nhất (điểm mạnh rõ nhất), (3) aspect có tỉ lệ negative/total cao nhất (điểm yếu rõ nhất), (4) aspect có `|positive - negative| / total` nhỏ nhất (ý kiến chia rẽ nhất) — chọn theo *tỉ lệ* chứ không theo số tuyệt đối để tránh việc luôn chọn trùng các aspect có tổng lượt nhắc lớn nhất cho cả 4 vai trò.

**Prompt.** Bảng 4 aspect đã chọn được serialize thành các dòng dạng `aspect=food | total=827 | positive=621 | negative=164 | neutral=42 | positive_reasons=great (109), good (100) | negative_reasons=mediocre (10), bad (7)`, kèm chỉ dẫn yêu cầu model viết **một đoạn văn phân tích tự nhiên** (không ép đúng một mẫu câu cố định), so sánh các aspect, chỉ được dùng đúng số liệu và reason đã cho, không được bịa thêm sự kiện/phần trăm/tên món ăn/thương hiệu nào khác (`build_reasoned_prompt` trong `src/report/flan_t5_report.py`).

**Model.** Nhóm fine-tune `google/flan-t5-base` trực tiếp trên chính định dạng bảng reason này (notebook Colab tự chứa `notebooks/finetune_flan_t5_reasoned_report_colab.ipynb`, dữ liệu train/valid là các bảng aspect tổng hợp), sinh bằng deterministic beam search (`num_beams=4`, `do_sample=False`) để kết quả tái lập được. Nếu factual checker (Mục 4.5) từ chối kết quả, hệ thống sinh lại tối đa 3 lần (không cần load lại model), thêm một câu nhắc "kết quả trước chưa đủ căn cứ, hãy sửa lại" vào đầu prompt cho các lần retry.

**CLI**: `python scripts/generate_report.py --model models/flan-t5-reasoned-report --output output/flan_t5_reasoned_report.json` (mặc định đọc `output/aspect_reasons_restaurant.json`, mảng `predicted` — bảng do chính T5-base ở Mục 4.2 sinh ra, không dùng gold, đúng kịch bản triển khai thực tế).

### 4.5. Factual checker

Vì báo cáo giờ là văn xuôi tự do (không theo một mẫu câu cố định), checker (`check_reasoned_report` trong `src/report/factual_checker.py`) không thể chỉ regex-match một khuôn câu như cách tiếp cận rule-based đơn giản ban đầu (Mục 2.5) — nhóm thực hiện theo 4 bước:

1. **Tách đoạn văn thành các mệnh đề** (theo dấu câu và các liên từ tương phản `whereas/while/but`), vì một câu có thể nói về nhiều aspect cùng lúc (ví dụ *"Indian food was praised... whereas dessert received 16 negative..."*).
2. **Gán mỗi mệnh đề cho đúng aspect** bằng khớp cụm từ, ưu tiên cụm dài hơn trước để tránh đếm nhầm substring (vd. `food` không được "ăn" luôn phần nhắc tới `Indian food`).
3. **Đối chiếu số liệu**: mọi con số xuất hiện trong mệnh đề của một aspect phải nằm trong tập số liệu *cho phép* của đúng aspect đó (total/positive/negative/neutral hoặc số đếm đi kèm reason phrase) — số liệu đúng nhưng gán nhầm sang aspect khác vẫn bị coi là sai.
4. **Đối chiếu reason**: mỗi aspect phải trích ít nhất một reason phrase thật sự thuộc về aspect đó trong bảng gốc; nếu đoạn văn dùng một reason phrase vốn thuộc aspect khác ("mượn" reason) thì bị gắn cờ `reasons belong to another aspect`.

Báo cáo chỉ được coi là `passed: true` khi cả 4 aspect bắt buộc đều xuất hiện, mọi mệnh đề hợp lệ, và không còn con số nào "trôi nổi" không gán được nguồn (`unattributed_numbers`).

---

## 5. Kết quả và đánh giá

### 5.1. So sánh baseline vs 3 model T5

| Model | Triplet-F1 | Precision | Recall |
|---|---|---|---|
| AsteLookupBaseline (phi-neural) | 0.3736 | 0.3222 | 0.4446 |
| T5-small | 0.7240 | 0.7238 | 0.7243 |
| **T5-base** | **0.7442** | 0.7609 | 0.7282 |
| FLAN-T5-base (lr=3e-4) | 0.5898 | 0.5910 | 0.5887 |
| FLAN-T5-base (lr=1e-4, thử lại) | 0.4159 | 0.4185 | 0.4133 |

*Bảng: so sánh trên cùng test split (1134 câu, 14res+15res+16res).*

![Dev triplet-F1 theo epoch, cả 4 lần chạy T5 (số liệu lấy trực tiếp từ log training trong notebooks-output/). t5-small và t5-base hội tụ và bão hoà rõ trong khoảng epoch 10–15; cả hai đường flan-t5-base vẫn đang tăng dần đều tới hết epoch 20, chưa hề bão hoà.](figures/triplet_f1_curves.png)

T5-base cải thiện gần gấp đôi triplet-F1 so với baseline tra cứu (0.3736 → 0.7442) và được chọn làm model chính. Biểu đồ trên cho thấy rõ lý do FLAN-T5-base thua kém: không phải nó "kém hơn" về bản chất mà đơn giản là chưa hội tụ trong ngân sách 20 epoch đã cấp cho cả 4 lần chạy. Xem Mục 6 để phân tích sâu hơn.

### 5.2. Chất lượng bảng tổng hợp aspect + lý do

Trên tập SemEval Restaurant có nhãn (4550 câu, train+dev+test), model T5-base cho **majority-sentiment agreement 97.25% (531/546)** giữa bảng tổng hợp từ gold triplets và bảng tổng hợp từ triplets model dự đoán (cùng một aspect, kết luận sentiment đa số có khớp không) — một mức tin cậy cao cho việc dùng bảng tổng hợp từ model làm input cho bước sinh báo cáo.

### 5.3. Demo quy mô lớn — Yelp Restaurant Reviews

Thay vì Amazon Reviews (dự kiến ban đầu, category Electronics — lệch domain hoàn toàn với model T5 train trên domain nhà hàng), nhóm đổi sang [Yelp Restaurant Reviews](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews) để domain khớp với model, không có gold label (chỉ demo quy mô lớn, không benchmark).

Chạy trên 2000 review mẫu (seed=42) → 14285 câu (sau khi tách câu từ review nhiều câu) → 14700 bộ ba → 1224 aspect (≥2 lượt nhắc). Mẫu lấy được nghiêng nhiều về nhóm quán tráng miệng/bakery: top aspect `ice cream` (765 lượt, positive), `place` (633), `flavors` (237), cùng `staff`/`service`/`donuts`/`pastries`/`bakery` — kết quả hợp lý, đúng domain.

### 5.4. Đánh giá factual checker

Trên lượt chạy demo thật (bảng `predicted` của `output/aspect_reasons_restaurant.json`, model FLAN-T5-base đã fine-tune trên định dạng reason), báo cáo sinh ra so sánh 4 aspect `food`/`indian food`/`dessert`/`waiter`, và **factual checker chấp nhận toàn bộ 4/4 claim (`passed: true`)** — không có aspect thiếu, không có số liệu "trôi nổi" không rõ nguồn. Đây là một lượt chạy demo minh hoạ (chưa phải một benchmark định lượng trên hàng trăm report), do khối lượng biên dịch báo cáo còn hạn chế trong thời gian làm đồ án.

Để bù lại, checker được kiểm chứng ở mức đơn vị khá kỹ: 16 test case trong `tests/test_report_generation.py` phủ các tình huống lỗi cụ thể — số liệu bịa (*hallucinated number*), thiếu một aspect bắt buộc, reason bị "mượn" từ một aspect khác, câu văn thừa không có căn cứ số liệu. Một ví dụ đối chứng rõ ràng: cùng một bảng thống kê track Laptop cũ (`output/aspect_stats.txt`), bản **chưa fine-tune** (`google/flan-t5-base` zero-shot, `output/flan_t5_natural_report.json`) bị checker từ chối (`accepted: false`) vì gán nhầm vai trò "mối lo ngại lớn nhất" cho `keyboard` thay vì `battery` và lặp `keyboard` hai lần; trong khi bản **đã fine-tune** trên đúng bảng đó (`output/flan_t5_finetuned_report.json`) sinh đúng 4/4 claim. Kết quả này cho thấy factual checker hoạt động đúng như thiết kế — bắt được lỗi thật khi model chưa được huấn luyện đủ, và không "quá tay" từ chối khi model đã bám sát nguồn.

### 5.5. Error analysis

Phân tích dưới đây dựa trực tiếp trên `output/aspect_reasons_restaurant.json` (bảng gold vs. predicted, T5-base, 4550 câu) và `output/aspect_reasons_yelp_demo.json` — không cần chạy lại model.

**(1) 15 case majority-sentiment lệch giữa gold và predicted đều là các aspect "sát nút", không lệch hệ thống.** Trong 546 aspect xuất hiện ở cả hai bảng, 531 khớp kết luận sentiment cuối cùng (97.25%, Mục 5.2); 15 case còn lại gần như luôn là những aspect có số lượt positive/negative gần bằng nhau, chỉ cần lệch 1 câu là đổi phe đa số — ví dụ `cheese` (gold: 3 positive/6 negative → negative; predicted: 5/4 → positive), `dish` (gold 6/7 → negative; predicted 8/7 → positive), `noodles` (gold 0/1 trên 2 câu → negative; predicted 1/1 → positive), `japanese food` (gold 3/4 → negative; predicted 4/3 → positive). Không có case nào lệch mạnh (ví dụ đa số rõ ràng ở gold nhưng đảo ngược hoàn toàn ở predicted) — cho thấy model không hề có thiên lệch sentiment hệ thống, sai số chỉ nằm ở các aspect vốn dĩ gây tranh cãi ngay cả với annotator.

**(2) Model bỏ sót nhiều aspect hiếm, và có xu hướng đếm thiếu (dưới) trên hầu hết aspect phổ biến — khớp với Precision > Recall đã thấy ở Mục 5.1.** 73 aspect chỉ có trong gold, không hề xuất hiện trong predicted — toàn bộ đều là aspect tần suất thấp (≤6 lượt nhắc, ví dụ `bathroom`, `flavor`, `ceiling`, `eggplant`), phần lớn tên món ăn hiếm/đặc thù mà model có thể chưa gặp đủ nhiều lúc train để tự tin trích ra. Ngược lại, trên các aspect phổ biến xuất hiện ở cả hai bảng, tổng lượt nhắc của predicted gần như luôn *thấp hơn* gold: `prices` (gold 69 → predicted 46, -23), `place` (278 → 258, -20), `food` (840 → 827, -13), `service` (568 → 562, -6). Đây là bằng chứng thứ hai, độc lập với Mục 5.1, cho cùng một hiện tượng: T5-base có Precision (0.7609) cao hơn Recall (0.7282) ở mức triplet — model "thận trọng", thà bỏ sót một số triplet mơ hồ còn hơn đoán bừa.

**(3) Aggregation theo chuỗi ký tự nguyên văn (exact-string) làm tách một aspect thành nhiều biến thể khác nhau — hạn chế của bước tổng hợp, không phải của model ASTE.** Ví dụ rõ nhất: gold có aspect `rice to fish ration` (lỗi chính tả có sẵn trong review gốc), trong khi predicted trích đúng chính tả `rice to fish ratio` — vì bước tổng hợp (Mục 4.3) gom nhóm theo `aspect.strip().lower()` nguyên văn, hai chuỗi chỉ khác 1 ký tự này bị tính là 2 aspect hoàn toàn tách biệt thay vì được gộp làm một. Hiện tượng tương tự lặp lại ở dạng số ít/số nhiều: track Restaurant có `place` (278 lượt, đã tính ở trên) tách biệt khỏi `places` (3 lượt, chỉ xuất hiện ở predicted); demo Yelp cho thấy mẫu hình này rõ hơn nhiều vì tập dữ liệu lớn hơn — `donuts` (189 lượt) tách khỏi `donut` (68 lượt), `cupcakes` (111) tách khỏi `cupcake` (59). Đây là hạn chế thật của cách tổng hợp (không lemmatize/chuẩn hoá hình thái), không phải lỗi của bước sinh triplet.

**(4) Aspect noise trong demo Yelp: `extract` (196 lượt, hạng 6) — kiểm tra lại giả thuyết ban đầu.** Ghi chú trước đó trong `plans/project-plan.md` suy đoán đây là phần bị cắt cụt từ "vanilla extract". Kiểm tra trực tiếp các reason phrase gắn với aspect `extract` cho thấy giả thuyết này **không đứng vững**: reason phổ biến nhất là `good`(8), `worth`(6), `try`(5), `best`(3) — hoàn toàn chung chung, không có phrase nào liên quan tới vani/hương liệu, và một số reason còn vô nghĩa ở dạng cô lập (`4.`, `huh`, `won't be coming back`). Nguyên nhân nhiều khả năng là model nhầm lẫn trên các câu ngắn/rời rạc, đặc thù của review Yelp quán tráng miệng (câu kiểu "Must try!", "Worth it.") khi bị tách câu — model không tìm được aspect thật trong câu và "bám" vào một token gần giống định dạng huấn luyện. Ghi chú trong `plans/project-plan.md` đã được cập nhật lại cho khớp với kết quả kiểm chứng này.

---

## 6. Khám phá mới / Thảo luận

**1. Đừng vội kết luận nguyên nhân từ một hyperparameter.** FLAN-T5-base thua kém rõ rệt so với T5-base cùng kích thước khi train với cùng lr=3e-4 (0.5898 vs. 0.7442 triplet-F1). Giả thuyết ban đầu: lr quá cao khiến loss ban đầu bất ổn (xấp xỉ 22, so với hai model kia ổn định ngay từ đầu). Thử lại với lr=1e-4 (giảm 10 lần) để kiểm chứng — nhưng kết quả *tệ hơn* (0.4159), và biểu đồ ở Mục 5.1 cho thấy `eval_triplet_f1` của cả hai lần chạy flan-t5-base vẫn tăng dần đều tới hết epoch 20, chưa hề bão hoà — trong khi t5-small/t5-base đã bão hoà rõ từ epoch 10–15. Kết luận đúng: lr thấp hơn chỉ làm hội tụ chậm hơn, không giải quyết được gì — nguyên nhân thực sự nhiều khả năng là FLAN-T5-base (đã instruction-tuned trên nhiều task đa dạng) cần nhiều epoch hơn hẳn để thích nghi lại với định dạng trích xuất terse này, không phải vấn đề learning rate.

**2. Baseline tra cứu: precision thấp hơn recall — dấu hiệu lỗi nằm ở bước ghép cặp, không phải bước tra cứu.** AsteLookupBaseline có Precision 0.3222 thấp hơn hẳn Recall 0.4446 — ngược với trực giác thông thường về một baseline tra cứu nguyên văn (thường kỳ vọng precision cao, recall thấp: cái gì đã "chắc chắn thấy" thì đúng, chỉ là bỏ sót nhiều). Khả năng cao nguyên nhân nằm ở bước ghép cặp "aspect gần opinion nhất theo khoảng cách token": trong câu có nhiều aspect/opinion xen kẽ, heuristic này dễ ghép nhầm một opinion với aspect không thực sự liên quan (dù bản thân aspect và opinion đều là cụm từ "đã biết" từ tập train), tạo ra nhiều bộ ba sai dạng false-positive. Đây là gợi ý tốt để làm error analysis chi tiết hơn ở bước tiếp theo (Mục 5.5).

---

## 7. Hạn chế

- **AsteLookupBaseline giới hạn từ vựng nghiêm ngặt**: chỉ khớp nguyên văn cụm từ đã thấy trong train, không có bất kỳ hình thức chuẩn hoá ngữ nghĩa nào (đồng nghĩa, biến thể hình thái). Recall 0.4446 phần lớn phản ánh giới hạn này hơn là độ khó thực sự của bài toán.
- **Không dùng POS tagging thật** cho baseline (chỉ tra cứu + ghép theo khoảng cách token) — một baseline rule-based "chuẩn" hơn trong literature thường dùng POS tagger để tìm ứng viên danh từ/tính từ trước khi ghép cặp; nhóm bỏ qua để tránh thêm dependency (NLTK/spaCy) không cần thiết cho một baseline tham chiếu.
- **Demo Yelp không có gold label** — độ tin cậy của bảng thống kê trên dữ liệu này chỉ được suy luận gián tiếp từ agreement 97.25% đo trên dữ liệu có nhãn cùng domain (SemEval Restaurant), không phải đo trực tiếp trên chính dữ liệu Yelp.
- **Factual checker mới kiểm chứng ở quy mô demo**: kết quả 4/4 claim hợp lệ (Mục 5.4) đến từ một lượt chạy minh hoạ, chưa phải benchmark định lượng trên hàng trăm report — cần chạy trên nhiều mẫu hơn để có số liệu % đáng tin cậy hơn (xem đề xuất ở Mục 8). Checker cũng vẫn là rule-based: dựa vào việc gán mệnh đề cho aspect bằng khớp cụm từ, nên có thể nhầm lẫn với các câu văn phức tạp hơn (nhiều aspect lồng nhau, câu dài nhiều mệnh đề phụ) so với các ví dụ demo hiện tại.
- **Bước tổng hợp gom nhóm theo chuỗi ký tự nguyên văn, không chuẩn hoá hình thái** (phát hiện ở error analysis, Mục 5.5): các biến thể chính tả/số ít-số nhiều của cùng một aspect (`rice to fish ration`/`rice to fish ratio`, `donuts`/`donut`) bị tính thành các aspect tách biệt trong bảng thống kê, làm phân mảnh số liệu và ảnh hưởng gián tiếp tới độ chính xác của báo cáo sinh ra.

---

## 8. Kết luận

Nhóm đã xây dựng một pipeline hoàn chỉnh sinh trực tiếp bộ ba khía cạnh–ý kiến–sentiment từ câu thô (AsteLookupBaseline → T5-small → T5-base, cải thiện triplet-F1 gần gấp đôi: 0.3736 → 0.7442), giải quyết đồng thời bài toán trích xuất aspect (không cần gán nhãn trước) và cho thêm trường *opinion* làm "lý do" phục vụ trực tiếp bước sinh báo cáo — với độ nhất quán tổng hợp cao (97.25% majority-sentiment agreement). T5-base là model được khuyến nghị dùng cho bước sinh báo cáo cuối cùng.

Bước sinh báo cáo và kiểm tra tính xác thực cũng cho một tín hiệu tích cực: trên lượt demo thật, FLAN-T5-base đã fine-tune sinh đúng 4/4 claim có căn cứ (Mục 5.4), và factual checker phân biệt được rõ ràng giữa model đã fine-tune (chấp nhận) và model chỉ prompt zero-shot (từ chối đúng lỗi gán nhầm vai trò) — cho thấy cả hai thành phần hoạt động đúng như thiết kế, dù mới kiểm chứng trên quy mô demo, chưa phải benchmark định lượng lớn.

Error analysis (Mục 5.5) không phát hiện thiên lệch hệ thống nào trong 3% case sentiment lệch — toàn bộ đều là các aspect vốn đã sát nút giữa gold và predicted — và xác nhận sai số chính của model nằm ở **recall trên aspect hiếm và đếm thiếu trên aspect phổ biến** (khớp với Precision > Recall đã đo ở Mục 5.1), chứ không phải ở việc suy luận sentiment sai. Phân tích cũng chỉ ra một hạn chế cụ thể, có thể sửa được ở bước tổng hợp (gộp aspect theo chuỗi ký tự nguyên văn thay vì theo hình thái), và đính chính một giả thuyết ban đầu sai (`extract` trong demo Yelp không phải do cắt cụt "vanilla extract" mà là model nhầm lẫn trên câu ngắn/rời rạc) — một ví dụ cho thấy tầm quan trọng của việc kiểm chứng giả thuyết bằng dữ liệu thật thay vì suy đoán.

**Hướng phát triển tiếp nếu có thêm thời gian**: (1) chuẩn hoá hình thái (lemmatize) ở bước tổng hợp aspect để gộp các biến thể chính tả/số ít-số nhiều, giảm phân mảnh số liệu; (2) thử fine-tune FLAN-T5-base cho ASTE với epoch budget lớn hơn 20 để kiểm chứng trực tiếp giả thuyết "cần nhiều epoch hơn" ở Mục 6; (3) chạy factual checker trên số lượng report lớn hơn để có số liệu tỉ lệ pass/fail đáng tin cậy thay vì một lượt demo; (4) cải tiến `AsteLookupBaseline` bằng POS tagging thật để tách rõ hạn chế do thiếu ngữ pháp khỏi hạn chế do từ vựng giới hạn.

---

## Phân công nhóm

| Việc | Thành viên |
|---|---|
| Data pipeline + baseline (Tuần 3) | Đạt |
| Fine-tune BERT/T5 + tổng hợp thống kê (Tuần 4) | Đạt, Sơn |
| FLAN-T5 sinh report + factual checker (Tuần 4) | Hoàng, Vinh, Hưng |
| Đánh giá & phân tích (Tuần 5) | **[TODO: phân công]** |
| Viết báo cáo + slides (Tuần 6) | **[TODO: phân công]** |

---

## Phụ lục

### A.1. Bảng hyperparameter đầy đủ

| Model | Epoch | LR | Batch (effective) | Seed |
|---|---|---|---|---|
| T5-small (ASTE) | 20 | 3e-4 | 16 | 42 |
| T5-base (ASTE) | 20 | 3e-4 | 16 | 42 |
| FLAN-T5-base (ASTE) | 20 | 3e-4 rồi 1e-4 | 16 | 42 |

*Hyperparameter khớp với config mặc định của repo nền Generative-ABSA [2] (`lr=3e-4`, `batch=16`, `epoch=20`).*

### A.2. Ví dụ output thực tế

Ví dụ một hàng trong bảng tổng hợp (`output/aspect_reasons_restaurant.json`), aspect `food` (827 lượt nhắc, T5-base):

> `positive` (621 lượt): lý do phổ biến nhất — `great`(109), `good`(100), `delicious`(39), `excellent`(37).
> `negative` (164 lượt): lý do phổ biến nhất — `mediocre`(10), `bad`(7), `overpriced`(5).

### A.3. Ví dụ báo cáo tự động sinh ra

Input (4 dòng trích từ `output/aspect_reasons_restaurant.json`, mảng `predicted`, rút gọn reason):

```
aspect=food | total=827 | positive=621 | negative=164 | neutral=42 | positive_reasons=great (109), good (100)
aspect=indian food | total=22 | positive=21 | negative=1 | neutral=0 | positive_reasons=great (5)
aspect=dessert | total=26 | positive=8 | negative=16 | neutral=2 | negative_reasons=not inspired (4)
aspect=waiter | total=42 | positive=24 | negative=18 | neutral=0 | positive_reasons=attentive (6) | negative_reasons=snobby (4)
```

Output của FLAN-T5-base (đã fine-tune) — `output/flan_t5_reasoned_report.json`:

> *"The strongest takeaway is the contrast between Indian food and dessert. Indian food was praised in 21 of 22 mentions, especially for being great, whereas dessert received 16 negative comments out of 26, largely citing not inspired. At the same time, food remained the most visible topic, appearing 827 times with 621 positive mentions and great as a recurring reason. Feedback on waiter was less settled: 24 positive versus 18 negative reactions across 42 mentions, with experiences ranging from attentive to snobby."*

Kết quả factual checker: `passed: true`, `claims_checked: 4`, `valid_claims: 4`, `missing_aspects: []`, `unattributed_numbers: []`.

---

## Tài liệu tham khảo

1. Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*. <https://alt.qcri.org/semeval2014/task4/>
2. Zhang, W., Li, X., Deng, Y., Bing, L., & Lam, W. (2021). *Towards Generative Aspect-Based Sentiment Analysis*. ACL-IJCNLP 2021. <https://github.com/IsakZhang/Generative-ABSA>
3. Peng, H., Xu, L., Bing, L., Huang, F., Lu, W., & Si, L. (2020). *Knowing What, How and Why: A Near Complete Solution for Aspect-Based Sentiment Analysis*. AAAI 2020.
4. Xu, L., Chia, Y. K., & Bing, L. (2020). *Position-Aware Tagging for Aspect Sentiment Triplet Extraction*. EMNLP 2020.
5. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. Journal of Machine Learning Research.
6. Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416.
7. Kale, M., & Rastogi, A. (2020). *Text-to-Text Pre-Training for Data-to-Text Tasks*. INLG 2020. <https://aclanthology.org/2020.inlg-1.14.pdf>
8. Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020). *Evaluating the Factual Consistency of Abstractive Text Summarization* (FactCC).
9. *Consistency Is the Key: Improving Factuality via Consistency-based Checking*. arXiv:2511.12236.
