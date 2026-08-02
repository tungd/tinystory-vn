# Báo cáo chi tiết — tinystories_v3

**Chủ đề:** Nghiên cứu *vị trí đặt LoRA adapter* (adapter placement) trên **SmolLM2-135M** cho bài toán sinh
truyện ngụ ngôn (moral fable) từ dataset **TF1-EN-3M**.
**Trọng tâm tài liệu này:** (1) xử lý dữ liệu, (2) phương pháp & phương pháp luận, (3) prompt engineering.
Đây là bản **chi tiết/kỹ thuật**; xem thêm bản dễ hiểu (dạng Q&A): [`bao-cao-de-hieu.md`](bao-cao-de-hieu.md).

> Ghi chú thuật ngữ: các từ kỹ thuật (LoRA, adapter, layer, module, perplexity, token, masking…) giữ nguyên
> tiếng Anh cho chính xác; phần diễn giải bằng tiếng Việt.

---

## 0. Câu hỏi nghiên cứu (một câu)

> Dưới cùng một ngân sách (cùng model nền, cùng rank, cùng dữ liệu, cùng lịch huấn luyện), **việc đặt LoRA adapter
> ở đâu có thay đổi chất lượng truyện không — và trục nào quan trọng hơn: các *layer* mà adapter phủ, hay các
> *module* mà adapter gắn vào?**

Để trả lời, ta huấn luyện 4 cấu hình và cô lập đúng 2 biến (xem §2.3).

---

## 1. Xử lý dữ liệu (data processing)

### 1.1 Nguồn dữ liệu
- **Dataset:** `klusai/ds-tf1-en-3m` (TF1-EN-3M) — **3 triệu** truyện ngụ ngôn tiếng Anh **sinh tổng hợp** bởi một
  model instruction ~8B. Giấy phép MIT.
- **Split:** `train` ≈ **2.8M** dòng · `validation` **100K** · `test` **100K**.
- **Các cột dùng đến:** `system_message` (chỉ dẫn hệ thống, cố định), `prompt` (đề bài có cấu trúc), `fable`
  (truyện đích). Các cột metadata khác (llm_name, host_gpu, thời gian…) không dùng cho huấn luyện.

### 1.2 Cấu trúc "đề bài" — 5 ô (5-slot)
Mỗi `prompt` trong dataset render ra **5 ô có nhãn**, model phải đan cả 5 vào truyện:

| Ô | Ý nghĩa | Ví dụ |
|---|---|---|
| **Main Character** | nhân vật chính (gộp cả trait) | *a clever skunk* |
| **Setting** | bối cảnh | *a flower field* |
| **Challenge** | xung đột trung tâm | *rivalry in love* |
| **Outcome** | cách giải quyết | *ancient enemies sign a pact* |
| **Teaching** | bài học đạo đức | *appearances can be deceiving* |

Ngoài 5 ô, `prompt` còn chứa các ràng buộc văn phong ("age group B (4–7 tuổi)", "vốn từ đơn giản", "bắt đầu bằng
cảnh", "không đặt tên nhân vật, dùng trait + loài", "kết bằng bài học", "khoảng 250 từ"…). **Ta giữ nguyên các
prompt này** (xem §3).

### 1.3 Chọn tập con (subsetting)
- Vì huấn luyện 4 arm và có giới hạn thời gian/GPU (1× Colab L4), ta **không** dùng cả 2.8M. Dùng **tập con cố định
  50.000 fable**, `seed=42`, **giống hệt nhau giữa các arm** (để so sánh công bằng).
- Cách lấy: `datasets.load_dataset(..., streaming=True).shuffle(seed=42, buffer_size=10_000).take(50_000)` →
  materialize. *Điểm cần nêu trung thực:* buffer 10k < 50k nên đây là "phần đầu stream được xáo nhẹ", không phải
  mẫu ngẫu nhiên đều — nhưng **xác định (deterministic) và giống nhau giữa các arm**, nên phép so sánh vẫn công bằng.
- **Tập đánh giá:** **500 dòng cố định** lấy từ `validation` (`seed=42`), tách hẳn khỏi train.

### 1.4 Định dạng mẫu huấn luyện — sinh có điều kiện + completion-only masking
Bài toán được mô hình hoá là **sinh có điều kiện (conditional generation)**: cho `system_message + prompt`, model
sinh `fable`. Code (`src/data.py`):

```python
def format_context(row):
    return f"{row['system_message'].strip()}\n\n{row['prompt'].strip()}\n\n"

def build_example(row, tokenizer, max_len=512):
    ctx_ids = tokenizer(format_context(row))["input_ids"]              # system + prompt (+ BOS)
    tgt_ids = tokenizer(row["fable"].strip(), add_special_tokens=False)["input_ids"] + [eos_id]
    input_ids = (ctx_ids + tgt_ids)[:max_len]
    labels    = ([-100]*len(ctx_ids) + tgt_ids)[:max_len]             # MASK phần ngữ cảnh
```

- **Completion-only masking:** mọi token của `system_message + prompt` bị gán nhãn `-100` (PyTorch bỏ qua khi tính
  loss). **Loss chỉ tính trên token của `fable`** (+ token kết thúc `<eot>`). Nhờ vậy model học *viết truyện*, không
  học *lặp lại đề bài*.
- **Token kết thúc:** thêm `<|endoftext|>` sau fable để model học **dừng đúng lúc**.
- **`max_seq_len = 512`** (prompt TF1 ~180 token + fable ~250 token < 512).
- **Tokenizer:** tokenizer gốc của SmolLM2 (vocab 49.152). Không train tokenizer mới.

**Vì sao completion-only masking quan trọng cho nghiên cứu này:** perplexity được tính **cùng một cách mask** cho
base và cả 4 arm → là tín hiệu **so sánh trực tiếp, công bằng** (cùng token, cùng cách chấm).

**Hàm loss & ý nghĩa của nhãn `−100`:**
- **Hàm loss = cross-entropy cho dự đoán token tiếp theo** (causal LM loss). Model sinh tự hồi quy (đoán từng token);
  loss = `−(1/N)·Σ log P(token_đúng | các token trước)`, chỉ trên N token của fable. Dùng cross-entropy vì đây là hàm
  chuẩn train language model, phạt nặng khi gán xác suất thấp cho từ đúng (tương đương maximum likelihood).
  **`perplexity = exp(loss)`** ⇒ loss thấp ⇔ perplexity thấp.
- **`−100` KHÔNG phải "trừ 100"** — nó là **mã đánh dấu (sentinel)**. PyTorch `CrossEntropyLoss` có
  `ignore_index=−100` (mặc định): token nào có nhãn `−100` thì **bị bỏ qua** khi tính loss (không gradient). Ta gán
  `−100` cho toàn bộ token đề bài → loss chỉ tính trên fable.
- **Vì sao là `−100`?** Vì token id luôn ≥ 0 nên `−100` chắc chắn **không trùng** với token thật nào → an toàn làm
  "cờ bỏ qua". Đây là quy ước mặc định của PyTorch (số âm khác cũng được nếu khai báo `ignore_index` tương ứng).

---

## 2. Phương pháp & phương pháp luận

### 2.1 LoRA (Low-Rank Adaptation) — nền tảng
Một lớp tuyến tính tính `h = W·x`. Full fine-tune cập nhật toàn bộ `W` (`d_out×d_in` tham số). LoRA **đóng băng**
`W`, chỉ học một cập nhật *hạng thấp*:

```
W' = W + ΔW,   ΔW = (α/r)·B·A,   A ∈ ℝ^(r×d_in),  B ∈ ℝ^(d_out×r),   B khởi tạo = 0
```

- Chỉ `A`, `B` được huấn luyện → **`r·(d_in+d_out)` tham số/ma trận**, ít hơn nhiều bậc.
- `B=0` lúc đầu ⇒ bắt đầu huấn luyện model **đúng bằng bản pretrain** (an toàn).
- Khi suy luận có thể **gộp** `BA` vào `W` ⇒ **không tốn thêm độ trễ** (ta dùng đúng tính chất này để export GGUF).

### 2.2 Vì sao chọn "model nhỏ pretrain + LoRA" (không train từ đầu, không full fine-tune model lớn)
- **Model nền:** `SmolLM2-135M` (base, không phải instruct) — kiểu Llama, **30 layer**, hidden 576, 9 head (3 KV,
  grouped-query), vocab 49.152.
- **Lý do chọn có tính quyết định:** SmolLM2 giữ **riêng** các projection `q_proj, k_proj, v_proj, o_proj` và
  `gate_proj, up_proj, down_proj`. Điều này khiến phép so sánh "chỉ thích nghi `q,v`" **xác định được**. Một model
  có **QKV hợp nhất** (vd `c_attn` của GPT-2) *không* tách được `q` khỏi `v` → không làm được ablation module.
- **Base (không instruct):** cho một phép **before/after trung thực** — base hoàn toàn không bám được prompt fable.

**So sánh với các lựa chọn khác (vì sao không chọn):**

| Lựa chọn | Vì sao KHÔNG chọn |
|---|---|
| GPT-2 124M | QKV hợp nhất (`c_attn`) → không tách q/v → **hỏng ablation module** |
| SmolLM2-360M | cũng tách rời nhưng lớn hơn 2,6× → chậm hơn, không cần |
| Qwen/Llama 1–4B | quá nặng cho ablation nhiều arm; là hướng của đồ án model-lớn (tinystory-vn) |
| Train từ đầu | tốn nhiều compute/thời gian; không tận dụng pretrain |

> **Tóm lại:** chọn SmolLM2-135M không phải vì "xịn nhất", mà vì nó **đúng công cụ cho câu hỏi nghiên cứu** — đủ nhỏ
> để chạy nhiều arm, và kiến trúc **tách rời** để tách được biến "module".

**7 projection (module) trong mỗi tầng — làm gì:**

| Ký hiệu | Thuộc | Chức năng | LoRA arm chỉnh |
|---|---|---|---|
| `q_proj` (query) | Attention | từ đang viết "hỏi" cần thông tin gì từ ngữ cảnh | A, B, **C** |
| `k_proj` (key) | Attention | mỗi từ trước "dán nhãn" để `q` so khớp | **C** |
| `v_proj` (value) | Attention | nội dung lấy về từ từ được chú ý | A, B, **C** |
| `o_proj` (output) | Attention | gộp kết quả attention, đưa ra ngoài | **C** |
| `gate_proj` | MLP | cổng lọc (SwiGLU) — nét nào được "bật" | **C** |
| `up_proj` | MLP | bung 576 → 1536 chiều để xử lý | **C** |
| `down_proj` | MLP | nén 1536 → 576 thành kết quả để viết | **C** |

- **q,k,v,o = cơ chế tra cứu ngữ cảnh** (q so k → chú ý → lấy v → o gộp) — giữ mạch truyện, tham chiếu đại từ.
- **gate,up,down = MLP xử lý** — nơi chứa phần lớn "kho câu chữ, giọng kể, mô-típ".
- **Vì sao ablation chọn q,v cho A/B:** theo bài báo LoRA gốc, adapt `Wq, Wv` thường là "đủ" — ta lấy làm baseline.
  **Arm C thêm o + toàn bộ MLP** để kiểm tra: chính MLP mới là chỗ tạo khác biệt lớn nhất (kết quả xác nhận).

### 2.3 Thiết kế ablation — cô lập đúng 2 biến
Giữ **cố định** `r=16, α=32, dropout=0.05` cho *mọi* arm ⇒ **vị trí đặt adapter là biến duy nhất**. 4 cấu hình:

| Arm | Adapter đặt trên | Layer | Cô lập |
|---|---|---|---|
| **base** | — (không fine-tune) | — | mốc nền |
| **A** | `q_proj, v_proj` | toàn bộ 30 | — |
| **B** | `q_proj, v_proj` | 10 layer cuối (index 20–29) | **layer** (vs A) |
| **C** | cả 7 projection tuyến tính | toàn bộ 30 | **module** (vs A) |

- **A vs B** = giữ nguyên module (`q,v`), chỉ đổi **độ sâu layer**.
- **A vs C** = giữ nguyên độ phủ layer (30), chỉ đổi **độ rộng module**.
- Có **unit test** (`tests/test_arms.py`) kiểm chứng: adapter arm B *thực sự* chỉ nằm ở layer 20–29, arm A phủ đủ
  30 → khẳng định "layer nào" không chỉ là danh nghĩa.

**Số tham số thích nghi** (r=16): A ≈ 0,92M (0,68%); B ≈ 0,31M (0,23%); C ≈ 4,88M (3,5% của 135M).

### 2.4 Cấu hình huấn luyện
AdamW · `lr=2e-4` · scheduler `cosine` · warmup 3% · `bf16` · batch 16 × grad_accum 2 (**hiệu dụng 32**) ·
**2 epoch** · seq 512 · **≈ 3.125 step/arm** · 1× Colab L4. Adapter mỗi arm (vài MB) đẩy lên HF Hub; chỉ số log về
Weights & Biases (project `tinystories_v3`) kèm heartbeat.

### 2.5 Phương pháp luận đánh giá (evaluation methodology)
Trên **500 dòng held-out cố định** (validation, seed 42), 3 tầng:

1. **Perplexity (chỉ số CHÍNH).** Teacher-forced, loss chỉ trên token fable (cùng cách mask như train), cộng dồn
   **có trọng số theo số token** (không phải trung bình ngây thơ theo dòng). Perplexity = `exp(mean cross-entropy)`.
   *Vì sao là chính:* không cần API, so sánh trực tiếp, khách quan, cùng thước đo cho mọi arm.
2. **Chỉ số reference-free** trên 100 bản sinh/arm: **Distinct-1/2** (đa dạng từ), **Self-BLEU** (trùng lặp; thấp =
   đa dạng), **Flesch Reading Ease** (độ dễ đọc; <0 = không đọc được).
3. **LLM-as-judge (ĐÃ CHẠY).** **1 judge cục bộ** = `Qwen2.5-7B-Instruct` (4-bit), chấm **4 trục** (grammar,
   creativity, moral_clarity, prompt_adherence) thang 1–10, **n=50 bản sinh/arm**. `overall` = trung bình 4 trục.
   *Chủ đích:* đơn giản hoá thành **1 judge** (không phải panel 3 judge của paper) — ghi trong ADR-0003.

**Nguyên tắc phương pháp luận:** kết luận dựa trên **thứ hạng** (ranking) do 2 thước đo **độc lập** (perplexity và
judge) cùng đưa ra, không dựa vào con số tuyệt đối của một chỉ số đơn lẻ.

**Perplexity (PPL) tính chính xác thế nào.** Đo *teacher-forced* — luôn cho model thấy token đúng phía trước, chỉ
chấm xác suất nó gán cho token kế tiếp:

```
# 1) Mỗi token fable thứ i: model cho phân phối p(·) trên 49.152 từ vựng
#    loss_token = −log p(token_đúng_i)                 (cross-entropy)
# 2) Gộp: trung bình CÓ TRỌNG SỐ theo số token, chỉ trên token fable
#    (token ngữ cảnh bị che = −100 nên KHÔNG tính vào mẫu số)
loss = ( Σ_i −log p(token_i) ) / N_token_fable
# 3) Perplexity:
PPL  = exp(loss)
```

*Diễn giải:* PPL = 3.84 ⇔ mỗi bước model "phân vân" như đang chọn giữa ~3,84 khả năng ngang nhau; PPL nhỏ ⇒ gán xác
suất cao cho từ đúng ⇒ nắm văn phong tốt hơn. Cộng dồn **theo token** ⇒ truyện dài không bị pha loãng; che ngữ cảnh
(−100) ⇒ PPL chỉ đo phần *sinh fable*, không tính phần đề bài.

**Tokenizer & khả năng so sánh PPL.** PPL tính *trên token*, mà mỗi tokenizer cắt cùng một câu thành số token khác
nhau ⇒ **PPL của hai model khác tokenizer KHÔNG so trực tiếp được** (mẫu số N_token khác bản chất):

| Hệ (trong báo cáo) | Loại tokenizer | Vocab | Ghi chú |
|---|---|---:|---|
| **E3 — SmolLM2-135M** (của em) | Byte-level BPE (kiểu Llama/GPT-2), có sẵn | 49.152 | Dùng nguyên, không train mới |
| E1 — SLM 60M (from-scratch) | BPE **tự train** (GPT-2-compat) | 12.000 | "May đo" cho domain fable |
| E2 — GPT 63M (from-scratch) | Metaspace BPE **tự train** | 16.384 | Thẻ char/moral/story riêng |
| E4/E5 — fine-tune 3B (Llama/Qwen) | tiktoken BBPE | 128k–152k | Vocab lớn, nén token gọn |
| Judge nội bộ — Qwen2.5-7B | Qwen BBPE (tiktoken) | ~152.000 | Chỉ để chấm ablation |
| Judge CHUNG — Gemma 4 26B | SentencePiece (unigram) | ~256.000 | Trọng tài liên nhóm |

⇒ **Trong E3** (base + 4 arm cùng tokenizer 49.152): xếp hạng bằng **perplexity** là hợp lệ. **Giữa 5 hệ E1–E5**
(3 loại tokenizer khác nhau): phải dùng **một LLM-judge chung (Gemma) chấm cùng bộ đề**, vì judge đứng ngoài mọi
tokenizer — đây là lý do nhóm cần judge chung.

### 2.6 Quy trình train từng bước & learning rate

**Quy trình (lặp cho mỗi arm A/B/C; base không train):**
1. Tải tokenizer + model base SmolLM2-135M (bf16).
2. Gắn LoRA theo cấu hình arm (`get_peft_model`) → "mở khoá" 0,3–4,9M tham số; **toàn bộ 135M gốc đóng băng**.
3. Lấy 50k fable (seed 42) → `build_example` (tokenize + mask ngữ cảnh `-100` + thêm `<eot>`, cắt 512).
4. `Trainer` (HuggingFace) + `DataCollatorForSeq2Seq` (pad + giữ mask); `save_strategy="no"`.
5. Train **2 epoch (~3.125 step)**: mỗi step forward → loss chỉ trên token fable → backward → **AdamW chỉ cập nhật
   tham số LoRA** (model gốc bất động).
6. Log loss/lr lên W&B (heartbeat ~50 step); lưu adapter + push HF Hub.
7. Thời gian ≈ 15–45 phút/arm trên 1× Colab L4.

**Learning rate:**
- Đỉnh `lr = 2e-4`, lịch **cosine + warmup 3%**:
  - ~94 step đầu (3% của 3125): lr **tăng** 0 → 2e-4 (warmup, tránh "sốc" gradient).
  - ~3.031 step sau: lr **giảm mượt theo cosine** về gần 0 (hội tụ êm).
- **Vì sao 2e-4?** Mức chuẩn cho LoRA (1e-4 → 3e-4), **cao hơn full fine-tune** (1e-5 → 5e-5), vì LoRA chỉ chỉnh ít
  tham số nên cần lr lớn hơn để học kịp. *(Xem W&B thấy lr thấp ở cuối = đuôi cosine, đúng thiết kế.)*
- **Optimizer:** AdamW (betas 0.9/0.999, weight decay 0).

**Bảng tham số đầu vào (đầy đủ):**

| Nhóm | Tham số | Giá trị |
|---|---|---|
| LoRA | r / alpha / dropout / bias | 16 / 32 / 0.05 / none |
| | target_modules | A,B: `q,v` · C: cả 7 linear |
| Data | subset / seq_len / epochs | 50.000 (seed 42) / 512 / 2 |
| Optim | optimizer / lr / scheduler / warmup | AdamW / 2e-4 / cosine / 0.03 |
| | precision / batch / grad_accum | bf16 / 16 / 2 (hiệu dụng 32) |
| Sinh (eval) | temp / top_p / rep_pen / max_new | 0.8 / 0.9 / 1.3 / 400 |

**Train loss thật (log trên W&B, 312 điểm/arm, mỗi 10 step):** cả 3 arm bắt đầu ≈ 2.27, tụt nhanh ~200 step đầu rồi
phẳng dần (đuôi cosine). **Loss cuối: C 1.35 < A 1.58 < B 1.71** — đúng thứ hạng perplexity. (Biểu đồ: bản HTML
`bao-cao.html` tab ⚙️; dữ liệu thô: `loss_history.json`.)

---

## 3. Prompt engineering

Điểm quan trọng cần nói rõ với thầy: **ta KHÔNG "chế" prompt mới cho huấn luyện.** Kỹ thuật prompt ở đây là (a) **tái
sử dụng trung thực** prompt có cấu trúc của TF1, và (b) **giữ nhất quán format giữa lúc train và lúc serve**, cộng
(c) thiết kế prompt cho **judge**. Cụ thể:

### 3.1 Prompt huấn luyện (train) = system_message + prompt (5 ô) của TF1
- **System message (cố định, dùng nguyên văn của dataset):**
  > *"You are a world-class creative assistant that generates captivating and morally-driven fables based on
  > structured inputs. Each fable must be: - Imaginative and coherent. - Appropriate for a wide audience, including
  > young readers. - Structured around a classic fable format (character, setting, conflict, resolution, and moral).
  > Age groups are defined as: A (≤3), B (4–7), C (8–11), D (12–15), E (≥16)."*
- **Prompt (5 ô + ràng buộc văn phong):**
  > *"Create a fable based on the following elements. Weave them naturally into a story: - Main Character: … -
  > Setting: … - Challenge: … - Outcome: … - Teaching: … The fable should: be appropriate for age group B (4–7),
  > use simple vocabulary, begin with vivid scene-setting, not use names (use trait + character), include simple
  > dialogue, show (don't tell) growth, end with a clear connection to the moral. Keep it around 250 words."*
- **Ghép:** `format_context = system_message + "\n\n" + prompt + "\n\n"`, target = `fable + <eot>`.

**Lý do (methodology):** giữ đúng phân phối prompt của dataset ⇒ model học đúng "văn phong đích"; và vì loss chỉ trên
fable nên prompt đóng vai **điều kiện điều khiển** (controllable generation qua 5 ô).

### 3.2 Prompt lúc phục vụ (serve) = Modelfile TEMPLATE của Ollama (khớp đúng train)
Khi export sang GGUF/Ollama, ta viết `Modelfile` sao cho **định dạng lúc chạy trùng với lúc train**:

```
TEMPLATE """{{ .System }}

{{ .Prompt }}

"""
SYSTEM """<đúng system_message của TF1 ở trên>"""
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
PARAMETER stop "<|endoftext|>"
```

- `TEMPLATE` tái tạo đúng `{{ .System }}\n\n{{ .Prompt }}\n\n` (giống `format_context`).
- `stop "<|endoftext|>"` khớp token kết thúc đã học lúc train.
- *Hạn chế đã lường trước:* app tinystory-vn có thể gửi prompt hơi khác câu chữ TF1 → model 135M (brittle) có thể
  yếu hơn số eval gốc (ghi trong ADR-0004).

### 3.3 Prompt cho LLM-judge (đánh giá)
Judge được ép **chỉ trả JSON**, chấm 4 trục kèm dẫn chứng (`src/judge.py`):

```
"You are a strict judge of children's fables. Given the REQUEST and the STORY, rate the STORY from 1 to 10 on
 four axes: grammar, creativity, moral clarity, prompt adherence. For EACH axis return an object with an integer
 "score" (1-10) and a short "reason" (one sentence, citing specific evidence quoted from the STORY).
 Respond ONLY with a JSON object using keys "grammar","creativity","moral_clarity","prompt_adherence".

 REQUEST:\n{prompt}\n\nSTORY:\n{story}\n\nJSON:"
```

- **Kỹ thuật prompt cho judge:** (1) yêu cầu **JSON-only** + đúng 4 key → dễ parse; (2) mỗi trục kèm `reason` **có
  trích dẫn từ truyện** → giảm chấm "trần trụi"; (3) đưa cả `REQUEST` (đề bài) + `STORY` để chấm được *prompt
  adherence*. Judge chạy **greedy** (do_sample=False) cho ổn định.
- **Chống parse lỗi:** hàm `parse_scores` chịu được JSON hỏng (trích số/regex fallback), thiếu trục → mặc định 0.

### 3.4 Tham số sinh (generation) khi tạo mẫu để đánh giá/serve
`temperature=0.8, top_p=0.9, repetition_penalty=1.3, max_new_tokens=400`, có **seed** để tái lập. Dùng chung cho cả
4 arm (công bằng).

---

## 4. Kết quả

### 4.1 Chỉ số tự động (perplexity + reference-free), 500 held-out

| Cấu hình | Val PPL ↓ | Distinct-1 | Distinct-2 | Self-BLEU | Flesch |
|---|---:|---:|---:|---:|---:|
| base | 9.52 | 0.557 | 0.971 | 0.007 | −66.2 |
| **A** — q,v·all-30 | 4.82 | 0.188 | 0.716 | 0.176 | **57.7** |
| **B** — q,v·last-10 | 5.46 | 0.190 | 0.739 | 0.171 | 51.1 |
| **C** — all-linear·all-30 | **3.84** | **0.210** | 0.728 | 0.191 | 52.8 |

Xếp hạng PPL: **C (3.84) < A (4.82) < B (5.46) ≪ base (9.52).**

### 4.2 LLM-as-judge (Qwen2.5-7B, n=50/arm, thang 1–10)

| Cấu hình | Grammar | Creativity | Moral clarity | Prompt adherence | **Overall** |
|---|---:|---:|---:|---:|---:|
| base | 6.68 | 5.24 | 5.92 | 5.08 | **5.73** |
| **A** | 6.90 | **7.16** | 7.12 | 5.62 | **6.70** |
| **B** | 6.02 | 6.54 | 6.40 | 4.78 | **5.94** |
| **C** | **7.36** | 6.94 | **7.16** | **6.00** | **6.87** |

Xếp hạng judge overall: **C (6.87) > A (6.70) > B (5.94) > base (5.73).**

> ⚠️ **Đây là judge NỘI BỘ (Qwen2.5-7B), chỉ để so base/A/B/C trong riêng E3 — KHÔNG so được với nhóm khác.**

### 4.2b Hai tầng đánh giá — và judge CHUNG của nhóm (Gemma 4 26B)

Báo cáo Nhóm 16 tách rõ **2 tầng, không trộn**:
1. **Judge nội bộ** (mỗi thành viên tự chấm; E3 dùng Qwen2.5-7B) → chỉ chọn cấu hình tốt nhất trong hướng của mình,
   **không dùng xếp hạng liên nhóm** (khác đề, giám khảo, thang đo).
2. **Judge CHUNG = Gemma 4 26B** (`gemma-4-26b-a4b-it`, temp 0): sinh lại đại diện mỗi hướng trên **cùng 25 đề**,
   chấm mù 4 trục — phép **so sánh liên nhóm công bằng**.

Dưới judge chung Gemma, đại diện E3 (`tsv3-smollm135-best` = LoRA C):

| Gemma judge · 25 đề | Ngôn ngữ | Sáng tạo | Moral | Bám đề | **Overall** |
|---|---:|---:|---:|---:|---:|
| E4 · Base+Repair (3B) | 10.0 | 7.40 | 9.40 | 10.0 | **9.20** |
| E5 · QLoRA (3B) | 9.88 | 7.04 | 8.56 | 8.28 | **8.44** |
| E1 · SLM 60M | 5.48 | 3.40 | 2.88 | 1.44 | **3.30** |
| E2 · V16 (63M) | 5.92 | 3.20 | 2.52 | 1.08 | **3.18** |
| **E3 · LoRA C (135M)** | 4.64 | 3.20 | 1.96 | 1.44 | **2.81** |

**6.87 (Qwen) và 2.81 (Gemma) KHÔNG mâu thuẫn** — hai judge, hai thang, hai mục đích. E3 thấp nhất ở vòng chung vì:
- **135M đấu với 3B** (E4/E5) — chênh cỡ mô hình ~24×;
- **runner chung BỎ `system_message`** mà E3 dùng khi train → **bám đề chỉ 1.44** (đúng cảnh báo prompt-format
  mismatch ở ADR-0004, giờ có số liệu xác nhận);
- giữa các model nhỏ: E3 ≈ E1 ≈ E2 (2.81/3.30/3.18; chênh E1–E3, E2–E3 gần như không có ý nghĩa thống kê) — khoảng
  cách lớn là **nhỏ-vs-3B**.

⇒ Đóng góp E3 là **"đặt adapter ở đâu"** (đo nội bộ), **không phải** thắng điểm tuyệt đối liên nhóm.

### 4.3 Nhận định
- **Hai thước đo độc lập ĐỒNG THUẬN:** perplexity và judge cho **cùng thứ hạng C > A > B > base** → kết luận vững.
- **Which modules (A vs C):** all-linear (thêm MLP) thắng — PPL 3.84 vs 4.82, và judge overall 6.87 vs 6.70 (C dẫn
  ở grammar, moral, adherence; A chỉ dẫn creativity). ⇒ **độ rộng module là đòn bẩy lớn nhất.**
- **Which layers (A vs B):** phủ toàn bộ layer thắng 1/3 layer cuối — PPL 4.82 vs 5.46, judge 6.70 vs 5.94. B đặc
  biệt yếu ở **prompt_adherence (4.78, còn thấp hơn base 5.08)**.
- **Fine-tune tăng mạnh chất lượng "thấy được":** creativity base 5.24 → A 7.16; moral 5.92 → 7.12 — những trục
  perplexity không nhìn thấy.
- **Best = C** (all-linear, toàn bộ layer), chỉ train **3,5%** trọng số; được export thành `tsv3-smollm135-best`.

---

## 5. Kết luận
Dưới ngân sách kiểm soát chặt, **vị trí đặt adapter thay đổi đo được** chất lượng. Đòn bẩy trội nhất là **độ rộng
module** (thêm adapter cho MLP), không phải độ sâu layer. Kết luận được **hai thước đo độc lập xác nhận cùng thứ
hạng**. Toàn bộ pipeline (data → train → eval → export GGUF/Ollama) tái lập được; code + adapter công khai.

**Tài liệu liên quan:** báo cáo học thuật đầy đủ `REPORT.md` · bản HTML `docs/report.html` · các quyết định
`docs/adr/0001..0004`.
