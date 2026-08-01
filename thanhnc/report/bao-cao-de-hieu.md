# Báo cáo dễ hiểu (Q&A) — tinystories_v3

Bản này viết theo kiểu **câu hỏi thầy có thể hỏi → trả lời ngắn gọn, dễ hiểu**, để ôn/trả lời vấn đáp.
Bản kỹ thuật đầy đủ: [`bao-cao-chi-tiet.md`](bao-cao-chi-tiet.md).

**Tóm tắt 1 câu:** Em lấy một model nhỏ có sẵn (**SmolLM2-135M**) rồi **dạy thêm bằng LoRA** để nó viết truyện ngụ
ngôn cho trẻ; câu hỏi nghiên cứu chính là **"nên gắn adapter vào layer/module nào của model thì tốt nhất?"**

---

## 0. Hiểu cơ bản: layer & module để làm gì (đọc phần này trước)

**Ví von:** model như một **dây chuyền 30 tầng**. Đề bài đi vào ở dưới, đi qua từng tầng — mỗi tầng "hiểu thêm một
chút" rồi "viết tiếp một chút" — cuối cùng ra thành truyện. Tầng dưới lo cái cơ bản (từ, ngữ pháp); tầng trên lo cái
trừu tượng (ý nghĩa, mạch, bài học).

**Trong MỖI tầng có 2 bộ phận (module):**

1. **Attention — "nhìn lại ngữ cảnh"** (gồm các ma trận `q, k, v, o`).
   Khi viết từ tiếp theo, nó quyết định *nên nhìn lại những từ nào đã viết* để giữ mạch (ai là ai, chuyện gì). Ví dụ
   viết "con cáo… **nó**…" thì attention giúp model biết "nó" = con cáo.
2. **MLP — "nghĩ ra câu chữ"** (gồm `gate, up, down`).
   Sau khi đã nhìn lại ngữ cảnh, đây là nơi model *nhào nặn* thành từ/cụm từ cụ thể — nó là **kho vốn từ + cách diễn
   đạt + mô-típ truyện** của model.

> "Đặt adapter ở đâu" = chọn **chỉnh những ma trận nào** (`q,v` hay cả `gate,up,down`…), **ở những tầng nào** (tất cả
> 30 hay chỉ 10 tầng cuối).

**7 bộ phận cụ thể trong mỗi tầng:**

| Ký hiệu | Thuộc | Việc của nó (dễ hiểu) |
|---|---|---|
| `q` (query) | Attention | Từ đang viết **hỏi**: "tôi cần thông tin gì từ các từ trước?" |
| `k` (key) | Attention | Mỗi từ trước **tự dán nhãn** mình chứa gì, để so khớp với `q` |
| `v` (value) | Attention | Nếu chú ý tới từ đó thì **lấy nội dung gì** mang về |
| `o` (output) | Attention | **Gộp** kết quả tra cứu rồi đưa ra ngoài |
| `gate` | MLP | **Cổng lọc**: nét thông tin nào được "bật" cho qua |
| `up` | MLP | **Bung** ra không gian lớn hơn để "nghĩ" (576 → 1536) |
| `down` | MLP | **Nén** lại thành kết quả để viết ra (1536 → 576) |

*Ví dụ:* viết "Con cáo thông minh… **Nó**…" → `q` của "Nó" khớp `k` của "con cáo" → lấy `v` = "cáo, thông minh" về
(`o` gộp) ⇒ giữ mạch. Rồi **MLP** (`up→gate→down`) nghĩ ra câu chữ tiếp theo đúng giọng fable.

**Ảnh hưởng:** arm A/B chỉnh `q,v` = sửa "cách nhìn ngữ cảnh" (loss 1.58/1.71); arm C chỉnh thêm **MLP** = sửa "kho
câu chữ" → tốt nhất (loss **1.35**). ⇒ chất fable nằm nhiều ở **MLP**.

**Tại sao "chỉnh" lại giúp model tốt hơn?**
Model gốc biết tiếng Anh chung chung nhưng **chưa quen văn phong truyện ngụ ngôn cho trẻ** (giọng kể, cấu trúc 5
phần, kết bằng bài học). LoRA **không sửa toàn bộ** model (dễ hỏng kiến thức gốc) mà **gắn thêm "núm chỉnh" nhỏ** vào
vài bộ phận, dạy chúng *nghiêng* sang văn phong fable. Chỉ dạy thêm 0,7%–3,5% trọng số, nhưng **chỉnh đúng chỗ** nên
đủ để đổi hẳn văn phong.

**Tại sao chỉnh MLP giúp nhiều nhất (vì sao C thắng)?**
Viết fable chủ yếu là chuyện **chọn đúng từ, đúng mô-típ, đúng giọng kể** — kho câu chữ đó nằm ở **MLP**. Chỉnh MLP =
chỉnh trực tiếp "cái đầu nghĩ câu chữ" → cải thiện mạnh nhất. Chỉ chỉnh attention (`q,v`) thì mới sửa "cách nhìn lại
ngữ cảnh", chưa động vào kho câu chữ → kém hơn.

**Tại sao chỉnh cả 30 tầng tốt hơn chỉ 10 tầng cuối (A > B)?**
Văn phong ngấm vào **cả dây chuyền**, không riêng mấy tầng cuối. Phủ cả 30 tầng rộng hơn nên tốt hơn; chỉ 10 tầng
cuối thì rẻ hơn nhưng bỏ sót → yếu hơn.

**Chốt một câu:** Attention = "đọc lại đề & ngữ cảnh"; MLP = "cái đầu nghĩ ra câu chữ". Vì fable cần **giọng kể & câu
chữ**, chỉnh vào **MLP trên toàn bộ tầng** (cấu hình C) là hiệu quả nhất.

---

## A. Tổng quan

**Q: Đề tài làm gì?**
A: Sinh **truyện ngụ ngôn tiếng Anh cho trẻ 4–7 tuổi** (có nhân vật, tình huống, bài học đạo đức). Điểm nghiên cứu:
khi dùng **LoRA** để fine-tune, **đặt adapter ở đâu** (layer nào, module nào) thì hiệu quả nhất.

**Q: Vì sao chọn model nhỏ 135M mà không train từ đầu hay dùng model lớn?**
A: Model nhỏ có sẵn + LoRA thì **rẻ, nhanh** (chạy vài phút/lần trên 1 GPU L4), đủ để **so sánh nhiều cách đặt
adapter**. Đây cũng là **góc khác** với bạn cùng nhóm (bạn ấy train model từ đầu / dùng model lớn), nên hai đồ án bổ
sung cho nhau.

**Q: "Adapter placement" nghĩa là gì?**
A: LoRA gắn các "miếng vá" (adapter) nhỏ vào model. Có 2 lựa chọn: **gắn vào lớp (layer) nào** và **gắn vào bộ phận
(module) nào** trong mỗi lớp. Em làm thí nghiệm để xem lựa chọn nào quan trọng.

---

## B. Xử lý dữ liệu

**Q: Dữ liệu lấy ở đâu?**
A: Dataset **TF1-EN-3M** trên HuggingFace — **3 triệu** truyện ngụ ngôn sinh sẵn. Chia sẵn: 2.8M train / 100K
validation / 100K test. Mỗi dòng gồm 3 phần dùng đến: `system_message` (chỉ dẫn), `prompt` (đề bài), `fable`
(truyện mẫu).

**Q: Đề bài (prompt) có dạng gì?**
A: 5 ô: **Main Character, Setting, Challenge, Outcome, Teaching** (nhân vật / bối cảnh / xung đột / kết cục / bài
học), kèm ràng buộc "viết cho trẻ 4–7 tuổi, ~250 từ".

**Q: Xử lý dữ liệu thế nào?**
A: 4 bước chính:
1. **Lấy tập con 50.000 truyện** (cố định, seed 42) — không cần cả 3 triệu; **dùng chung một tập cho cả 4 thí
   nghiệm** để so sánh công bằng.
2. **Ghép** `system_message + prompt` làm đầu vào, `fable` làm đầu ra.
3. **Che phần đề bài** khi tính loss (masking): model chỉ bị "chấm điểm" trên phần **viết truyện**, không phải phần
   chép lại đề → nó học *viết*, không học *nhại đề*.
4. Thêm dấu **kết thúc** để model biết dừng; cắt độ dài tối đa 512 token.

**Q: Vì sao phải "che đề bài" (masking)?**
A: Để model tập trung học **viết truyện**. Và nhờ mọi thí nghiệm dùng **cùng cách che**, chỉ số **perplexity** so
sánh được công bằng.

**Q: "Gán nhãn −100" là gì? Tại sao trừ 100?**
A: **Không phải "trừ 100"** — `−100` là một **mã đánh dấu** (cờ) nghĩa là "**bỏ qua ô này khi tính loss**". Đây là
giá trị mặc định của PyTorch (`ignore_index=−100`). Ta gán `−100` cho mọi token của đề bài → loss chỉ tính trên phần
truyện. Chọn số −100 vì token thật luôn ≥ 0 nên −100 **không bao giờ trùng** với token thật → an toàn làm cờ.

---

## C. Phương pháp

**Q: LoRA là gì (nói đơn giản)?**
A: Thay vì sửa **toàn bộ** trọng số của model (tốn kém), LoRA **đóng băng** model gốc và chỉ học thêm **vài ma trận
nhỏ** gắn vào. Rất ít tham số, nhanh, và có thể **gộp lại** vào model để chạy không chậm hơn.

**Q: Em thí nghiệm những gì? (4 cấu hình)**
A:
- **base** = model chưa fine-tune (mốc so sánh).
- **A** = gắn adapter vào `q,v` (một phần của attention) ở **tất cả 30 lớp**.
- **B** = gắn `q,v` nhưng **chỉ 10 lớp cuối**.
- **C** = gắn vào **cả 7 bộ phận tuyến tính** (cả attention + MLP) ở **tất cả 30 lớp**.

**Q: Tại sao đặt đúng 4 cấu hình này?**
A: Để **tách riêng 2 câu hỏi**:
- **A so với B** → chỉ khác *số lớp* → trả lời "**layer nào**".
- **A so với C** → chỉ khác *số bộ phận* → trả lời "**module nào**".
Giữ nguyên mọi thứ khác (rank=16…) nên khác biệt kết quả **chỉ do vị trí đặt adapter**.

**Q: Vì sao chọn SmolLM2-135M cụ thể?**
A: Vì nó **tách rời** các bộ phận `q, k, v, o, gate, up, down`, nên em mới **so sánh được** "chỉ gắn q,v". Model kiểu
GPT-2 gộp q,k,v làm một khối → không tách được → không làm thí nghiệm này được.

**Q: Huấn luyện thế nào?**
A: 2 epoch, learning rate 2e-4, cosine schedule, trên 1 GPU Colab L4, mỗi cấu hình vài phút–vài chục phút. Adapter
lưu lên HuggingFace.

**Q: Quy trình train cụ thể từng bước?**
A: (1) tải model base; (2) **gắn LoRA** (đóng băng model gốc, chỉ mở khoá vài triệu tham số); (3) lấy 50k truyện,
tokenize + che phần đề bài; (4) train ~3.125 step — mỗi step chỉ cập nhật phần LoRA; (5) log lên W&B; (6) lưu adapter
+ đẩy lên HuggingFace. Lặp cho A, B, C; base không train.

**Q: Learning rate 2e-4 nghĩa là gì? Vì sao cosine + warmup?**
A: Learning rate = "bước học" mỗi lần cập nhật. `2e-4` (0.0002) là **mức chuẩn cho LoRA** — cao hơn khi fine-tune
toàn bộ model (vì LoRA chỉnh ít tham số nên cần bước lớn hơn để học kịp). **Warmup 3%**: mấy chục step đầu tăng dần
từ 0 (tránh "sốc"), rồi **cosine** giảm mượt về gần 0 ở cuối (hội tụ êm). *(Xem W&B thấy lr thấp ở cuối là bình
thường — đó là đuôi cosine.)*

**Q: Vì sao không dùng GPT-2 hay model lớn hơn?**
A: **GPT-2 không được** vì nó **gộp q,k,v làm một khối** → không tách được để so sánh "chỉ q,v" → hỏng thí nghiệm.
**Model lớn (1–4B)** thì quá nặng để chạy nhiều thí nghiệm, và đó là hướng của bạn kia. SmolLM2-135M vừa **nhỏ/nhanh**
vừa **tách rời các bộ phận** → đúng công cụ cho câu hỏi "đặt adapter ở đâu".

---

## D. Prompt engineering

**Q: Em có tự viết prompt không? Prompt engineering ở đâu?**
A: Em **không "chế" prompt mới cho huấn luyện** — dùng **nguyên văn** system_message + đề bài 5 ô của dataset (để
model học đúng "văn phong đích"). Phần "prompt engineering" nằm ở 3 chỗ:
1. **Giữ đúng format** giữa lúc train và lúc chạy: khi đưa lên Ollama, em viết `Modelfile` sao cho template
   (`{{ .System }}` rồi `{{ .Prompt }}`) **trùng khớp** cách ghép lúc train, và đặt dấu dừng `<|endoftext|>` đúng
   như đã học.
2. **Điều khiển bằng 5 ô:** vì che phần đề bài khi tính loss, đề bài đóng vai "công tắc" điều khiển truyện — người
   dùng đổi 5 ô là đổi được truyện.
3. **Prompt cho judge (chấm điểm):** ép model chấm **chỉ trả JSON**, gồm 4 điểm (grammar/creativity/moral/adherence)
   **kèm câu lý do có trích dẫn từ truyện**, và đưa cả đề bài + truyện để chấm được "bám đề".

**Q: Tại sao Modelfile phải khớp format lúc train?**
A: Vì model đã học theo một khuôn cụ thể; nếu lúc chạy đưa sai khuôn thì nó viết kém đi. Đây là bài học quan trọng về
"train sao — serve vậy".

---

## E. Đánh giá

**Q: Đánh giá bằng gì?**
A: 3 tầng:
1. **Perplexity (chính)** — đo model "bất ngờ" đến đâu với truyện thật; **thấp hơn = tốt hơn**. Khách quan, không cần
   API, so sánh trực tiếp được.
2. **Chỉ số văn bản** — Distinct (đa dạng từ), Self-BLEU (trùng lặp), **Flesch** (độ dễ đọc).
3. **LLM-as-judge** — dùng một model khác (**Qwen2.5-7B**) chấm 50 truyện/cấu hình theo 4 tiêu chí, thang 1–10.

**Q: Hàm loss dùng là gì? Tại sao?**
A: **Cross-entropy cho dự đoán token tiếp theo** (chuẩn của mọi model sinh văn bản). Model đoán từng từ một; loss đo
"model gán xác suất cho **từ đúng** cao đến đâu" — gán thấp thì bị phạt nặng. Dùng cross-entropy vì nó tương đương
**tối đa hoá xác suất dữ liệu thật**. Điểm hay: **perplexity = exp(loss)**, nên loss giảm thì perplexity giảm. Biểu đồ
loss (xem bản HTML, tab ⚙️) tụt nhanh lúc đầu rồi phẳng dần.

**Q: Perplexity là gì (nói đơn giản)?**
A: Mức độ model "ngạc nhiên" khi thấy truyện đúng. Càng ít ngạc nhiên (perplexity thấp) nghĩa là model **nắm được
văn phong** càng tốt.

**Q: Chỉ dùng 1 judge, có bị thiên vị không?**
A: Có rủi ro (nên bài báo gốc dùng 3 judge). Em **chủ đích đơn giản hoá thành 1 judge** và **không kết luận theo con
số tuyệt đối**, mà theo **thứ hạng**. May mắn là **judge và perplexity cho cùng thứ hạng** → kết luận đáng tin.

**Q: Vì sao Flesch của base âm (−66)?**
A: Vì model base viết **văn bản lộn xộn/không đọc được** cho đề này → điểm dễ đọc âm. Sau fine-tune lên +51…+58
(mức "dễ đọc", hợp trẻ em).

---

## F. Kết quả

**Q: Kết quả chính là gì?**
A: Xếp hạng chất lượng: **C > A > B > base**, và **cả perplexity lẫn judge đều cho cùng thứ hạng này.**

| | base | A (q,v·all) | B (q,v·10 lớp cuối) | C (all-linear) |
|---|---:|---:|---:|---:|
| Perplexity ↓ | 9.52 | 4.82 | 5.46 | **3.84** |
| Judge overall ↑ | 5.73 | 6.70 | 5.94 | **6.87** |

**Q: Kết luận về "layer nào / module nào"?**
A:
- **Module quan trọng hơn:** gắn thêm adapter cho **MLP** (cấu hình C) cho kết quả tốt nhất — đây là **đòn bẩy lớn
  nhất**.
- **Layer:** phủ **toàn bộ layer** tốt hơn chỉ 10 lớp cuối (A > B). Chỉ 10 lớp cuối rẻ hơn nhưng yếu hơn rõ, nhất là
  độ "bám đề".
- **Tốt nhất = C**, mà chỉ cần huấn luyện **3,5%** trọng số của model.

**Q: Fine-tune giúp gì rõ nhất?**
A: Tăng mạnh **sáng tạo** (creativity 5.24 → 7.16) và **độ rõ bài học** (moral 5.92 → 7.12) — những thứ perplexity
không đo được, nhưng judge thì thấy.

---

## G. Điểm yếu & câu hỏi khó

**Q: Hạn chế của đồ án?**
A:
- Model chỉ 135M → **chất lượng tuyệt đối** không bằng model lớn (đề tài chỉ so *tương đối* về vị trí đặt adapter).
- **Chạy 1 lần/1 seed**, chưa có khoảng tin cậy.
- **Chưa đủ ô 2×2** (thiếu cấu hình all-linear × 10-lớp-cuối).
- **1 judge** (không phải panel).
- Tập con 50k **xáo trộn nhẹ** (buffer nhỏ), tuy giống nhau giữa các arm nên vẫn công bằng.

**Q: Nếu làm lại sẽ cải thiện gì?**
A: Thêm nhiều seed (để có sai số), thêm ô 2×2 còn thiếu, quét thử rank khác nhau, và dùng panel nhiều judge.

**Q: So với đồ án model lớn (tinystory-vn) khác gì?**
A: Bên kia **train từ đầu / model lớn**, tập trung vào **ứng dụng + phương pháp đánh giá**. Bên em **model nhỏ +
LoRA**, tập trung vào **câu hỏi đặt adapter ở đâu**. Model tốt nhất của em (C) đã được đóng gói (GGUF) và cắm vào app
của bên kia để so sánh trực tiếp trong "Compare mode".

**Q: Con số perplexity của em có so được với bên model-from-scratch không?**
A: **Không so trực tiếp được**, vì hai bên dùng **bộ từ vựng (tokenizer) khác nhau** — perplexity phụ thuộc tokenizer.
Muốn so công bằng phải dùng **cùng một thước đo bên ngoài**, ví dụ **cùng một LLM-judge chấm cùng đề** (đó là vai trò
của judge trong app).
