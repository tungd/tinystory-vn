# Nên đặt adapter ở đâu? Nghiên cứu vị trí đặt LoRA adapter trên SmolLM2-135M cho bài toán sinh truyện ngụ ngôn

> **Ghi chú:** Đây là báo cáo của dự án đồng hành **tinystories_v3**, được port vào `tinystory-vn` (chỉ tài liệu). Toàn bộ **code, ADR, test và notebook** được nhắc tới nằm ở repo gốc: https://github.com/harryct229/tinystories_v3 — các đường dẫn tương đối như `src/`, `docs/adr/`, `tests/`, `notebooks/` là của repo đó. Bản tiếng Anh: [`report.md`](report.md) · bản HTML: [`report.vi.html`](report.vi.html).

**Dự án:** `tinystories_v3` · **Môn:** Generative AI · **Loại:** Báo cáo kỹ thuật
**Model nền:** SmolLM2-135M · **Dataset:** TF1-EN-3M (`klusai/ds-tf1-en-3m`) · **Phần cứng:** 1× Colab L4 · **Ngày:** 2026-07-13
**Code:** `github.com/harryct229/tinystories_v3` · **Adapter:** `hf.co/congthanh991/tsv3-smollm135-{A-qv-all,B-qv-last3,C-alllinear}`

---

## Tóm tắt

Dòng nghiên cứu *TinyStories* đã chứng minh rằng các mô hình ngôn ngữ nhỏ hơn nhiều so với một tỷ tham số vẫn có
thể sinh ra truyện thiếu nhi mạch lạc nếu được huấn luyện trên dữ liệu tổng hợp hẹp và chất lượng cao. Dựa trên tiền
đề đó, chúng tôi lấy một model **SmolLM2-135M** đã pretrain và fine-tune bằng **LoRA** (Low-Rank Adaptation) trên
**TF1-EN-3M** — kho ba triệu truyện ngụ ngôn có bài học đạo đức được sinh tổng hợp — và đặt một câu hỏi làm trọng
tâm nghiên cứu: **nên đặt các adapter low-rank ở đâu.** Giữ nguyên rank, dữ liệu và lịch huấn luyện, chúng tôi so
sánh một mốc nền (không fine-tune) với ba cách đặt adapter — `q,v` của attention trên **toàn bộ layer** (**A**),
cùng cấu hình đó nhưng chỉ ở **1/3 layer cuối** (**B**), và **cả 7 projection tuyến tính** trên toàn bộ layer
(**C**). Trên tập held-out cố định, perplexity giảm từ **9.52** (base) xuống **3.84** (C), và độ dễ đọc Flesch tăng
từ **−66** (gần như không đọc được) lên **+53** (phù hợp lứa tuổi). Hai phép so sánh một-biến trả lời đúng tiêu đề:
đặt adapter trên *toàn bộ* layer thắng cấu hình 1/3 layer cuối (4.82 so với 5.46), nhưng phần thắng lớn nhất đến từ
**độ rộng module** — thêm adapter cho các projection **MLP** (3.84 so với 4.82). Một LLM judge 4 trục (n = 50 mỗi
arm) tái lập đúng thứ hạng đó từ những giả định độc lập — **C 6.87 > A 6.70 > B 5.94 > base 5.73** — và làm rõ một
điểm mà perplexity đọc sai: cấu hình 1/3 layer cuối không phải lựa chọn "rẻ mà gần bằng" như nó tỏ ra, vì prompt
adherence của nó (4.78) tụt xuống *dưới* cả model chưa huấn luyện (5.08). Dự án được đặt như một **góc tiếp
cận đối lập có chủ đích** với dự án đồng hành vốn pretrain model của mình từ đầu; nó đánh đổi chất lượng tuyệt
đối để đổi lấy một góc nhìn có kiểm soát về hiệu quả đặt adapter, và được triển khai đầy đủ: huấn luyện, đánh giá,
export sang GGUF, và phục vụ qua Ollama ngay trong ứng dụng của dự án đồng hành.

---

## 1 · Giới thiệu

### 1.1 Động cơ

Truyện ngụ ngôn (fable) là một dạng tự sự cô đọng và có cấu trúc chặt: một nhân vật kèm tính cách, một bối cảnh, một
thử thách, một cách giải quyết, và một bài học rõ ràng ở cuối. Chính sự đều đặn đó khiến nó là một bài kiểm tra tốt
bất thường cho các model nhỏ — từ vựng hạn chế, mạch truyện ngắn, và chất lượng dễ đánh giá bằng mắt người đọc.
*TinyStories* [2] cho thấy các model dưới 10M tham số vẫn viết được truyện thiếu nhi mạch lạc khi dữ liệu được tuyển
chọn kỹ và đủ hẹp; dataset chúng tôi dùng [1] được xây dựng có chủ đích để "thuận lợi cho việc parameter-efficient
fine-tuning các model nhỏ ở hạ nguồn".

Thay vì huấn luyện từ đầu, chúng tôi khởi đầu từ một model nhỏ **đã pretrain** và thích nghi nó bằng **LoRA** [3] —
kỹ thuật đóng băng trọng số nền và chỉ học một số ít ma trận cập nhật low-rank chèn vào các projection được chọn.
Vị trí đặt LoRA thường được chọn theo quy ước. Đề bài của dự án biến quy ước đó thành đóng góp — *"ghi lại vì sao ta
chọn thêm layer nào vào model"* — mà chúng tôi hiểu chính xác là **vị trí đặt adapter (adapter placement)**.

> **Câu hỏi nghiên cứu.** Dưới cùng một ngân sách (cùng model nền, rank, dữ liệu và lịch huấn luyện), **việc đặt
> adapter ở đâu có thay đổi chất lượng truyện không — và trục nào quan trọng hơn: các *layer* mà adapter phủ, hay
> các *module* mà adapter gắn vào?**

**Đóng góp.** (1) Một ablation có kiểm soát, đổi một-biến, về vị trí đặt LoRA cho bài toán sinh tự sự trên một model
135M, tách riêng trục *độ sâu layer* (A vs. B) và trục *độ rộng module* (A vs. C); (2) một kết quả định lượng —
**độ rộng module lấn át độ sâu layer** cho bài toán này; (3) **hai dụng cụ đo độc lập trên cùng bốn arm** —
perplexity teacher-forced và một LLM judge 4 trục — đồng thuận về thứ hạng và bất đồng, một cách hữu ích, về việc
giới hạn layer tốn kém đến đâu; (4) một pipeline hoàn chỉnh, tái lập được, kèm bước export GGUF → Ollama để cắm vào
một ứng dụng đồng hành.

---

## 2 · Nền tảng & lý thuyết

### 2.1 Low-Rank Adaptation (LoRA)

Một lớp tuyến tính tính `h = W·x`. Full fine-tune cập nhật toàn bộ `d_out · d_in` phần tử của `W`. LoRA đóng băng
`W` và học một cập nhật **low-rank**:

```
W' = W + ΔW ,   ΔW = (α / r) · B · A ,   A ∈ ℝ^(r × d_in) ,  B ∈ ℝ^(d_out × r)
```

với rank `r ≪ min(d_in, d_out)`, `B` khởi tạo bằng 0 (nên lúc bắt đầu huấn luyện model đúng bằng bản pretrain), và
hệ số co giãn cố định `α/r`. Chỉ `A` và `B` được huấn luyện, cho `r·(d_in + d_out)` tham số mỗi ma trận được thích
nghi — ít hơn `d_in·d_out` rất nhiều bậc. Khi suy luận, tích `BA` có thể được **gộp (merge)** trở lại vào `W`, nên
model đã gộp không tốn thêm độ trễ nào — đây chính là cách chúng tôi export một model đặc duy nhất sang GGUF (§8).

### 2.2 Vì sao "đặt ở đâu" là một câu hỏi thật

LoRA để ngỏ hai lựa chọn, và đó chính là hai trục của nghiên cứu này:

- **Module nào?** Một block gồm các projection attention (`q,k,v,o`) và các projection MLP (`gate,up,down`). Bài báo
  LoRA cho thấy chỉ thích nghi `W_q`, `W_v` thường là đủ — nhưng "thường đủ" là một khẳng định cần kiểm chứng theo
  từng bài toán, không phải một định luật.
- **Layer nào?** Adapter có thể phủ mọi layer hoặc chỉ một tập con; các layer sau thường được lập luận là mang biểu
  diễn đặc thù-bài-toán nhiều hơn, tạo động lực cho lựa chọn "chỉ 1/3 layer cuối" như một đòn bẩy hiệu quả.

Hai trục này thường bị trộn lẫn trong thực tế. Thiết kế ở §5.3 **tách chúng ra có chủ đích** để mỗi trục đọc được
riêng.

### 2.3 Vì sao dạng truyện ngụ ngôn hợp với model nhỏ

Văn phong đích hẹp có chủ đích: truyện ngụ ngôn ~200 từ cho trẻ, cấu trúc 5 phần cố định, từ vựng hạn chế. Một model
135M không đủ dung lượng và cũng không cần mô hình hoá văn bản mở; chính sự hẹp của bài toán khiến model nhỏ khả thi
và khiến câu hỏi vị trí đặt adapter trở nên sắc nét — có tín hiệu thật để nắm bắt, và những khác biệt nhỏ về *nơi*
thêm dung lượng trở nên nhìn thấy được trong các chỉ số.

---

## 3 · Dataset

Chúng tôi huấn luyện trên **TF1-EN-3M** (`klusai/ds-tf1-en-3m`) [1], ba triệu truyện ngụ ngôn tiếng Anh sinh tổng
hợp bởi một model instruction 8 tỷ tham số. Mỗi dòng ghép một **prompt có cấu trúc** với một **fable** và một
**system message** cố định. Prompt hiển thị năm ô có nhãn mà model phải đan vào truyện:

| Ô | Ý nghĩa | Ví dụ |
|---|---|---|
| **Main Character** | nhân vật chính (character + trait gộp lại) | *a clever skunk* |
| **Setting** | bối cảnh câu chuyện | *a flower field* |
| **Challenge** | xung đột trung tâm | *rivalry in love* |
| **Outcome** | cách giải quyết | *ancient enemies sign a pact* |
| **Teaching** | bài học | *appearances can be deceiving* |

Các split: **2.8M train / 100K validation / 100K test**. Chúng tôi tái sử dụng dataset và bộ tiêu chí đánh giá của
bài báo thay vì tự chế. Một cặp huấn luyện minh hoạ — đưa ra để cố định *văn phong đích*, không phải đầu ra của
model:

> "In a sun-kissed flower field, a clever skunk loved to sniff out the sweetest blooms … As they bent to drink, they
> saw their reflections in the calm water. 'We've been judging each other wrong,' said the skunk … From that day on
> they promised to look beyond appearances — a reminder that true beauty comes from within."

---

## 4 · Định vị: khác gì so với `tinystory-vn`

Dự án này là một góc đối lập có chủ đích với dự án đồng hành `tinystory-vn`, vốn nhắm cùng dataset và văn phong đích
nhưng từ đầu kia của không gian thiết kế. Dự án đó **tự dựng prior từ đầu**: một decoder kiểu Llama 30M pretrain
trên TF1 với tokenizer 12k riêng, sau đó là chiến dịch năm phương pháp post-training (DPO, SFT-on-best, RAFT,
GRPO-lite, distillation) chạy dưới cùng một protocol LLM-judge cố định, và cuối cùng là model 60M huấn luyện trên
toàn bộ kho 2.34 triệu truyện — model được chốt cho ứng dụng. Đóng góp của họ là đường cong huấn luyện from-scratch
đó, chiến dịch có kiểm chứng (bốn kết quả null và một kết quả âm, được báo cáo như phát hiện chính danh) và ứng dụng
đã triển khai (guardrail, judge, chế độ Compare), trong đó Qwen3-4B vừa là mốc tham chiếu model lớn vừa là giám khảo.

Chúng tôi xuất phát từ tiền đề ngược lại: **kế thừa một prior thay vì tự dựng**, và biến **câu hỏi nội tại của model
về vị trí đặt adapter** thành trọng tâm — adapter nên gắn vào layer nào, module nào. Trục đó chỉ tồn tại khi model
nền bị đóng băng, nên về mặt cấu trúc nó vắng mặt trong một dự án from-scratch. Hai hướng bổ sung cho nhau; §7.6 đặt
các góc cạnh nhau.

---

## 5 · Phương pháp

### 5.1 Model nền — và vì sao chọn SmolLM2-135M

Chúng tôi khởi đầu từ **SmolLM2-135M** (bản base, không phải instruct) [4]: một decoder kiểu Llama với **30 layer**,
hidden size **576**, **9 attention head** (3 KV head, grouped-query attention), MLP intermediate 1536, vocab 49.152
token. Lựa chọn này mang tính quyết định cho nghiên cứu:

- **Projection tách rời.** SmolLM2 giữ riêng các ma trận `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`. Đây là điều khiến phép so sánh kinh điển "chỉ thích nghi `q,v`" *xác định được*. Một model
  có **QKV hợp nhất** (ví dụ `c_attn` của GPT-2, gộp `q,k,v` vào một ma trận) không tách được `q` khỏi `v`, khiến
  trục module không thể so sánh. (Xem ADR-0001.)
- **Base, không phải instruct.** Bắt đầu từ checkpoint base cho một phép before/after trung thực: model base hoàn
  toàn không bám được prompt fable, nên cả hiệu ứng fine-tune lẫn các khác biệt về vị trí đặt adapter đều nhìn thấy
  rõ so với một mốc nền thật.

### 5.2 Định dạng bài toán

Chúng tôi mô hình hoá việc học như **sinh có điều kiện (conditional generation)**. Đầu vào là `system_message ⧺
prompt`; đích là `fable`. Điểm mấu chốt: cross-entropy loss chỉ tính trên **các token của fable** — mọi token ngữ
cảnh (system + prompt) bị mask thành chỉ số bỏ qua `−100` (completion-only masking) — và một token kết thúc được
thêm vào để model học cách dừng:

```
input  =  BOS  s_1 … s_m  (system+prompt)   f_1 … f_n  <eot>
labels = −100 −100 … −100  (ngữ cảnh bị mask)  f_1 … f_n  <eot>
loss   =  −(1/N) · Σ_{t: label_t ≠ −100}  log p(label_t | x_<t)
```

Cách này giữ cho model điều khiển được lúc suy luận (câu chuyện được lái bằng năm ô) và, quan trọng cho một nghiên
cứu vị trí đặt adapter, khiến **perplexity trở thành một tín hiệu công bằng, cùng-điều-kiện** — mọi arm đều được
chấm với cùng một cách mask trên cùng một văn bản.

### 5.3 Thiết kế ablation — layer nào, module nào

Rank cố định ở **`r = 16`** (`α = 32`, dropout 0.05) xuyên suốt các arm để **vị trí đặt adapter là biến duy nhất**.
Bốn cấu hình được huấn luyện và đánh giá:

| Arm | Adapter đặt trên | Layer | Cô lập trục |
|---|---|---|---|
| **base** | — (không fine-tune) | — | mốc nền tham chiếu |
| **A** | `q_proj, v_proj` | toàn bộ 30 | — |
| **B** | `q_proj, v_proj` | 10 layer cuối (index 20–29) | *độ sâu layer* (vs. A) |
| **C** | cả 7 projection tuyến tính | toàn bộ 30 | *độ rộng module* (vs. A) |

- **A vs. B** giữ nguyên tập module (`q,v`) và chỉ đổi **độ sâu layer**.
- **A vs. C** giữ nguyên độ phủ layer (toàn bộ 30) và chỉ đổi **độ rộng module**.

Hai phép so sánh một-biến sạch — chính là lý do các arm được đặt như vậy. (Xem ADR-0002.) Chúng tôi kiểm chứng bằng
code rằng adapter của arm B thực sự chỉ nằm ở layer 20–29 và của arm A phủ cả 30, nên khẳng định "layer nào" không
chỉ là danh nghĩa (`tests/test_arms.py`).

### 5.4 Hạch toán số tham số được thích nghi

Với một ma trận `(d_in, d_out)`, LoRA ở rank `r` thêm `r·(d_in + d_out)` tham số. Với các chiều của SmolLM2:

| Projection | shape (in→out) | tham số @ r=16 |
|---|---|---:|
| `q_proj` | 576 → 576 | 18.432 |
| `k_proj` | 576 → 192 | 12.288 |
| `v_proj` | 576 → 192 | 12.288 |
| `o_proj` | 576 → 576 | 18.432 |
| `gate_proj` | 576 → 1536 | 33.792 |
| `up_proj` | 576 → 1536 | 33.792 |
| `down_proj` | 1536 → 576 | 33.792 |

So với model nền ≈134,5M: **A** (q,v × 30) ≈ 0,92M (0,68%); **B** (q,v × 10) ≈ 0,31M (0,23%); **C** (cả-7 × 30) ≈
4,88M (3,5%).

### 5.5 Cấu hình huấn luyện

Mỗi arm huấn luyện trên **tập con 50.000 fable cố định** (seed 42, giống nhau giữa các arm) trong **2 epoch** ở độ
dài chuỗi 512: AdamW, learning rate `2e-4`, lịch cosine với 3% warmup, bf16, batch hiệu dụng 32 (≈ 3.125 step/arm).
Huấn luyện trên một **Colab L4**; adapter mỗi arm (vài MB) được đẩy lên Hugging Face Hub, chỉ số stream về Weights &
Biases kèm một heartbeat callback để theo dõi.

### 5.6 Phương pháp đánh giá

Trên một **tập held-out 500 dòng cố định** của validation (seed 42), chúng tôi báo cáo:

- **Validation perplexity** (chính): teacher-forced, loss chỉ trên token fable (cùng cách mask như lúc huấn luyện),
  có trọng số theo số token; cùng cách chấm cho base và mọi arm. Đây là tín hiệu tự nhiên, không cần API, so sánh
  trực tiếp được — hợp nhất cho một nghiên cứu vị trí đặt adapter.
- **Chỉ số không tham chiếu (reference-free)** trên 100 bản sinh mỗi arm (temp 0.8, top-p 0.9, rep-pen 1.3, có seed):
  **Distinct-1/2** (đa dạng từ vựng), **Self-BLEU** (trùng lặp nội-tập; thấp = đa dạng hơn), **Flesch Reading Ease**
  (cao = dễ đọc hơn; dưới 0 = gần như không đọc được).
- **LLM-as-judge** — một judge cục bộ theo 4 trục của bài báo (grammar, creativity, moral clarity, prompt
  adherence), mỗi trục thang 1–10, **overall** là trung bình cộng bốn trục. Judge là **Qwen2.5-7B-Instruct** nạp
  4-bit (bitsandbytes) trên Colab L4, giải mã greedy, rubric chỉ trả JSON (`src/judge.py`, `src/run_judge.py`).
  Chấm **n = 50** bản sinh mỗi arm (50 dòng đầu của chính tập held-out 500 dòng), mỗi bản chấm theo đúng prompt
  yêu cầu của nó, cùng cấu hình sinh ở trên (tối đa 400 token mới, có seed). Đây là **một** judge chứ không phải
  panel (xem ADR-0003), và cố ý **không** phải judge Qwen3-4B ở tầng ứng dụng dùng cho so sánh liên model: lượt
  chấm này chỉ chấm bốn arm trong ablation của chúng tôi, nên các con số so được với nhau trong nội bộ và không
  được đem so với điểm do một judge khác sinh ra.

---

## 6 · Kết quả

Cả bốn cấu hình, đánh giá đồng nhất trên cùng 500 prompt held-out:

| Cấu hình | Tham số thích nghi | **Val PPL ↓** | Distinct-1 | Distinct-2 | Self-BLEU | Flesch |
|---|---:|---:|---:|---:|---:|---:|
| **base** (không FT) | 0 | 9.52 | 0.557 | 0.971 | 0.007 | −66.2 |
| **A** — `q,v` · all-30 | ≈ 0,9M | 4.82 | 0.188 | 0.716 | 0.176 | **57.7** |
| **B** — `q,v` · last-10 | ≈ 0,3M | 5.46 | 0.190 | 0.739 | 0.171 | 51.1 |
| **C** — all-linear · all-30 | ≈ 4,9M | **3.84** | **0.210** | 0.728 | 0.191 | 52.8 |

Xếp hạng theo perplexity: **C (3.84) < A (4.82) < B (5.46) ≪ base (9.52).**

### 6.1 LLM-as-judge

Cùng bốn cấu hình đó, chấm bởi judge 4 trục cục bộ (n = 50 mỗi arm, §5.6):

| Cấu hình | Grammar | Creativity | Moral clarity | Prompt adherence | **Overall ↑** |
|---|---:|---:|---:|---:|---:|
| **base** (không FT) | 6.68 | 5.24 | 5.92 | 5.08 | 5.73 |
| **A** — `q,v` · all-30 | 6.90 | **7.16** | 7.12 | 5.62 | 6.70 |
| **B** — `q,v` · last-10 | 6.02 | 6.54 | 6.40 | 4.78 | 5.94 |
| **C** — all-linear · all-30 | **7.36** | 6.94 | **7.16** | **6.00** | **6.87** |

Xếp hạng theo judge overall: **C (6.87) > A (6.70) > B (5.94) > base (5.73)** — *đúng thứ tự* của perplexity, từ một
dụng cụ đo không chia sẻ giả định nào với nó. Hai quan sát mà bảng perplexity không đưa ra được:

- Phần tăng lớn nhất nhờ fine-tune nằm ở **creativity** (5.24 → 7.16) và **moral clarity** (5.92 → 7.12), đúng những
  trục mà một chỉ số likelihood không nhìn thấy. Grammar gần như không đổi (6.68 → 7.36) vì model nền vốn đã trôi
  chảy; thứ nó thiếu là **hình thức** của thể loại.
- **Prompt adherence của B (4.78) tụt xuống dưới cả base chưa huấn luyện (5.08)**, ô duy nhất trong bảng mà
  fine-tune làm mọi thứ xấu đi. §7.2 bàn tiếp.

Con số nổi bật:

- **−60%** perplexity so với mốc nền chưa huấn luyện, cho cấu hình tốt nhất (C).
- **3,5%** trọng số của model được huấn luyện để đạt kết quả đó.
- **+119 điểm Flesch** (−66 → +53) — từ không đọc được đến phù hợp lứa tuổi.
- **+1.14 điểm judge** overall (5.73 → 6.87), với hai dụng cụ đo độc lập đồng thuận về thứ hạng.

---

## 7 · Phân tích & thảo luận

### 7.1 Fine-tune có tác dụng — rất mạnh

Model base chưa huấn luyện không bám được prompt: PPL 9.52 và Flesch âm mạnh. Distinct-1 **cao** (0.557) và Self-BLEU
**gần 0** (0.007) *không* phải tín hiệu chất lượng — chúng là dấu vân tay của văn bản **ngẫu nhiên, không ràng buộc**,
vốn tránh lặp một cách tầm thường. Mọi arm LoRA đều gói sự ngẫu nhiên đó lại thành fable mạch lạc, đúng khuôn. Một
lời cảnh báo đi kèm: các chỉ số đa dạng không-tham-chiếu, nếu đọc tách biệt, có thể tưởng thưởng cho sự thiếu mạch
lạc.

> **[Layer]** **A vs. B — đặt adapter trên toàn bộ layer thắng 1/3 layer cuối (PPL 4.82 vs. 5.46).** Độ phủ layer có
> ích. Riêng trên perplexity, khoảng cách trông khá dễ tha thứ: B lấy lại ~85% phần cải thiện của A so với base
> trong khi chỉ thích nghi *một phần ba* số layer (≈ 0,3M vs. ≈ 0,9M tham số), đọc như một đánh đổi hiệu quả hợp lý.

> **[Layer · judge nói ngược]** **Judge không đồng ý, và chỉ ở đúng chỗ này.** B đạt 5.94 so với 6.70 của A và thua
> trên *mọi* trục (grammar 6.02 vs. 6.90, creativity 6.54 vs. 7.16, moral clarity 6.40 vs. 7.12, adherence 4.78 vs.
> 5.62). Prompt adherence của B là con số duy nhất trong cả nghiên cứu **thấp hơn cả base chưa huấn luyện** (4.78 vs.
> 5.08): giới hạn adapter ở 10 layer cuối không chỉ lấy được ít phần cải thiện hơn, mà dường như còn **làm mất** khả
> năng bám prompt. Cách hoà giải: lấy lại 85% khoảng cách perplexity **không phải** là lấy lại 85% khoảng cách chất
> lượng. Perplexity lấy trung bình trên mọi token, nên một model có thể đoán tốt phần lớn token thông thường của một
> truyện mà vẫn hỏng ở số ít token phải tôn trọng 5 slot của prompt. Đánh đổi hiệu quả **đứng vững ở khía cạnh mô
> hình hoá, không đứng vững ở chất lượng sinh ra**; một báo cáo chỉ dựa vào perplexity sẽ khuyến nghị B trên một tiền
> đề sai.

> **[Module]** **A vs. C — all-linear thắng đậm attention-only (PPL 3.84 vs. 4.82, thấp hơn ≈ 20%).** Phần thắng lớn
> nhất của cả nghiên cứu đến từ việc thêm adapter cho các projection **MLP**, chứ không phải từ phủ thêm layer. Với
> bài toán này, **độ rộng module** quan trọng hơn **độ sâu layer** — khả dĩ vì các sublayer MLP mang phần lớn dung
> lượng cho các mẫu bề mặt/từ vựng mà một bài sinh hẹp như viết fable dựa vào. Judge xác nhận cùng chiều nhưng nhẹ
> hơn: C dẫn A ở **ba trên bốn trục** (grammar 7.36 vs. 6.90, moral clarity 7.16 vs. 7.12, adherence 6.00 vs. 5.62),
> overall 6.87 vs. 6.70; A giữ được một trục là **creativity** (7.16 vs. 6.94).

### 7.4 Cấu hình tốt nhất, và một đánh đổi

**C** (all-linear, toàn bộ layer) tốt nhất trên **cả hai dụng cụ đo** — PPL 3.84 và judge overall 6.87 — và cũng
giàu từ vựng nhất (Distinct-1 0.210); nó được export thành `tsv3-smollm135-best`. Một điểm tinh tế: **A** đạt Flesch
(độ dễ đọc) cao nhất (57.7 vs. 52.8) và dẫn ở trục creativity của judge (7.16 vs. 6.94) — model all-linear viết văn
đặc hơn, quy củ hơn một chút trong khi mô hình hoá bài toán tốt hơn. Với văn phong đích sư phạm, khả năng mô hình hoá
của C cùng phần dẫn ở grammar, moral clarity và adherence là kết quả tiêu điểm; A là á quân nhẹ ký đáng nhớ khi tài
nguyên khan hiếm, và là lựa chọn thú vị hơn nếu thứ cần tối ưu là độ đa dạng sáng tạo.

### 7.5 Kiểm tra định tính

Chạy hai model đã export trên một prompt chưa gặp (*a brave little turtle · a quiet pond · a sudden storm · the
animals work together · teamwork overcomes fear*) cho thấy khác biệt rõ. Model **base** lạc đề, sinh ra lời bình
kiểu hướng dẫn viết luận. Model **best (C)** viết một truyện ngụ ngôn thật:

> "In the quiet pond, where water lilies swayed gently in the breeze and fish swam happily by, a brave little turtle
> lived among his friends… One day, dark clouds gathered over the pond as strong winds howled and loud thunderclaps
> rumbled… The clever rabbit, strong and swift, had been watching from a nearby rock. He suggested that they work
> together to save their friends… The two groups of creatures huddled closer and began working together…"

Vài lỗi nhỏ (thi thoảng "(Figure 1)", nhân vật hơi lệch) là điều dễ hiểu với một model 135M đã lượng tử hoá Q8 và độ
lệch định dạng prompt nói ở §9, nhưng đầu ra rõ ràng là một fable mạch lạc, bám prompt.

### 7.6 So sánh liên dự án

| Góc | tinystories_v3 (dự án này) | tinystory-vn |
|---|---|---|
| Prior | **kế thừa** — SmolLM2-135M pretrained | **tự dựng từ đầu** — 30M, rồi 60M trên full TF1 |
| Cách thích nghi | LoRA; ablation vị trí đặt adapter (4 arm) | đường cong pretrain + 5 phương pháp post-training |
| Câu hỏi nghiên cứu | *đặt* capacity huấn luyện được ở đâu | SLM from-scratch đẩy được tới đâu |
| Tham số được huấn luyện | 3.5% của 135M (≈ 4.9M) | 100% của 30M / 60M |
| Ngân sách huấn luyện | 100k mẫu fable, vài phút/arm trên 1× L4 | 934M token, 10.000 bước (60M), Colab T4 |
| Metric chính | perplexity val (teacher-forced, có mask) | LLM-judge, 4 trục, seed bắt cặp, n = 45 |
| Khả năng điều khiển | prompt có điều kiện 5 slot | prompt có điều kiện 5 slot |
| Bàn giao | adapter GGUF đăng ký trong app của họ | app FastAPI + React đầy đủ + guardrail + judge |

Bản demo dự kiến rất trực diện: adapter tốt nhất (và base) được gộp, chuyển sang GGUF, và đăng ký trong danh sách
model Ollama của `tinystory-vn`, để chế độ **Compare** hiện có đặt bản fine-tune 135M của chúng tôi cạnh cả mốc 4B
(lớn hơn ≈ 30×) lẫn model 60M from-scratch của họ. Thực tế judge trong app là thước đo công bằng *duy nhất* giữa hai
dự án: perplexity không so được với nhau khi một bên dùng BPE 12k tự huấn luyện còn một bên dùng vocab 49k của
SmolLM2, trong khi judge chấm cùng bộ prompt qua cùng một dụng cụ đo. (Xem ADR-0004.)

---

## 8 · Bàn giao: GGUF → Ollama → tinystory-vn

Kết quả nghiên cứu được bàn giao dưới dạng một model chạy được. Với **base** và **best (C)**: **(1) gộp** adapter C
vào base (`merge_and_unload`); **(2) chuyển** sang **GGUF Q8_0** qua `llama.cpp` (≈ 138 MB mỗi bản); **(3)** viết một
**Modelfile** Ollama có `TEMPLATE` tái tạo đúng định dạng huấn luyện và dừng ở `<|endoftext|>`; **(4)** `ollama
create` cả hai model; **(5)** thêm hai mục vào `config/models.json` của `tinystory-vn`. Cả hai model đã được tạo và
kiểm chứng **khác nhau về hành vi** (base lạc đề; best viết fable). Chế độ Compare khi đó đặt bản 135M cạnh model 4B.
(Xem ADR-0004.)

---

## 9 · Hạn chế & mối đe doạ với tính hợp lệ

- **Trần chất lượng tuyệt đối.** 135M giới hạn chất lượng; nghiên cứu bàn về hiệu quả đặt adapter *tương đối*, không
  nhằm vượt một model lớn.
- **Chạy một lần, không có khoảng tin cậy.** Mỗi arm một seed/lịch; ước lượng điểm không kèm confidence interval (dù
  các khoảng cách PPL cách nhau thoải mái).
- **2×2 chưa đầy đủ.** Ô còn thiếu `all-linear × last-third` sẽ cho phép kiểm tra tương tác giữa hai trục.
- **Một judge, không panel, chưa đo nhiễu.** Judge 4 trục chỉ là một model (Qwen2.5-7B-Instruct, 4-bit) ở n = 50 mỗi
  arm, nên nó mang theo mọi thiên lệch của model đó, và chúng tôi **chưa** đo nhiễu giữa các lần chấm bằng cách chấm
  lại cùng một arm hai lần. Hãy đọc từng điểm số như chỉ báo; **thứ hạng** mới là tín hiệu bền, và đó chính là phần
  đồng thuận với perplexity. Mức đồng thuận giữa các judge khác họ model vẫn chưa được đo. Cũng lưu ý judge khá dễ
  dãi với grammar của base (6.68 cho văn bản mà Flesch coi là không đọc được), nên overall của base (5.73) không thấp
  như khoảng cách perplexity gợi ý.
- **Điểm judge không mang đi nơi khác được.** Các con số này đến từ judge và rubric của riêng chúng tôi; không so
  được với điểm của judge Qwen3-4B ở tầng ứng dụng hay của bất kỳ nghiên cứu nào khác, chỉ so được giữa bốn arm.
- **Lệch định dạng prompt ở hạ nguồn.** Adapter huấn luyện trên đúng câu chữ của TF1 có thể chấm thấp hơn khi bị điều
  khiển bằng prompt hơi khác của tinystory-vn. Chấp nhận có chủ đích (ADR-0004).
- **Xáo trộn dữ liệu nhẹ.** Tập con 50k lấy qua streaming shuffle với buffer 10k — xác định và giống nhau giữa các
  arm (nên so sánh vẫn công bằng), nhưng gần với "phần đầu được xáo nhẹ" hơn là một mẫu đều.

---

## 10 · Tính tái lập

- **Code:** `github.com/harryct229/tinystories_v3` — phần logic được unit-test trên CPU; train/eval có smoke test
  model tí hon. Chạy `python -m pytest`.
- **Adapter:** công khai trên Hub — `congthanh991/tsv3-smollm135-{A-qv-all, B-qv-last3, C-alllinear}`.
- **Train / eval:** `notebooks/colab_runner.ipynb` (hoặc `python -m src.train --arm {A,B,C} --push`), rồi các cell
  eval; kết quả trong `results_auto.json`.
- **Lượt chấm judge:** `src/judge.py` + `src/run_judge.py`; điểm thô trong `results_judge.json`, bản chuẩn trên Hub
  tại `congthanh991/tsv3-smollm135-eval/results_judge.json`.
- **Hai ghi chú cho lần chạy lại trên Colab** (ghi lại để khỏi tái diễn): (i) Colab cài sẵn `torchao 0.10.0` khiến
  bước tiêm LoRA của PEFT báo lỗi ở một kiểm tra phiên bản — `pip uninstall -y torchao` (ta không dùng nó); (ii)
  websocket kernel của Colab CLI không ổn định với lệnh dài — chạy huấn luyện như một **job nền** và theo dõi tiến độ
  qua Hub (adapter xuất hiện), đừng tail log.

---

## 11 · Kết luận & hướng phát triển

Dưới một ngân sách kiểm soát chặt, **vị trí đặt adapter thay đổi đo được** mức độ một model nhỏ học viết fable tốt
đến đâu. Phủ *toàn bộ* layer thắng 1/3 layer cuối, nhưng đòn bẩy trội nhất là **độ rộng module** — gắn adapter vào
các projection MLP, không chỉ attention, mới cho kết quả tốt nhất (PPL 3.84, giảm 60% so với mốc nền) trong khi chỉ
huấn luyện 3,5% trọng số của model. Một model nhỏ đã pretrain cộng với các adapter low-rank đặt khéo là đủ để biến
đầu ra không đọc được thành fable phù hợp lứa tuổi.

Đứng sau câu trả lời đó là **hai dụng cụ đo, không phải một**: một LLM judge 4 trục tái lập đúng xếp hạng của
perplexity (C 6.87 > A 6.70 > B 5.94 > base 5.73) mà không chia sẻ giả định nào với nó. Chỗ hai bên tách nhau tự nó
là một kết quả: perplexity tô hồng cấu hình 1/3 layer cuối, thứ mà judge cho điểm bám prompt còn *thấp hơn* model
chưa huấn luyện. **Một chỉ số likelihood lấy trung bình trên mọi token có thể bỏ sót một thất bại tập trung ở số ít
token mang bài toán** — đó là bài học phương pháp chúng tôi mang sang nghiên cứu đặt adapter tiếp theo.

**Hướng phát triển:** đo nhiễu của chính judge bằng cách chấm lại một arm hai lần, và thêm một judge khác họ model,
trước khi coi bất kỳ khoảng cách dưới một điểm nào là thật; quét rank để tách hiệu ứng vị trí khỏi dung lượng thêm;
bổ sung ô `all-linear × last-third` để hoàn thiện 2×2 và kiểm tra tương tác; và dùng các model đã export cho một so
sánh định lượng với các model của dự án đồng hành dưới judge ở tầng ứng dụng.

---

## Tài liệu tham khảo

1. Nadás, Dioșan, Piscoran, Tomescu (2025). *TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small,
   Open Language Models.* arXiv:2504.20605. Dataset: `klusai/ds-tf1-en-3m`.
2. Eldan & Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
3. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
   arXiv:2106.09685.
4. Allal et al. (2025). *SmolLM2.* `HuggingFaceTB/SmolLM2-135M`, Hugging Face Hub.
5. `tinystory-vn` — dự án đồng hành (SLM 30M/60M from-scratch trên TF1, chiến dịch post-training có kiểm chứng,
   app đã triển khai với Qwen3-4B làm mốc tham chiếu kiêm giám khảo). `github.com/tungd/tinystory-vn`.

---

## Phụ lục A · Cấu hình chính xác

```
model nền        HuggingFaceTB/SmolLM2-135M   # kiểu Llama, 30 layer, hidden 576, 9/3 head, vocab 49152
LoRA (cố định)   r=16  alpha=32  dropout=0.05  bias=none  task=CAUSAL_LM
  arm A          target=[q_proj,v_proj]                     layers=toàn bộ 30
  arm B          target=[q_proj,v_proj]                     layers=20..29 (1/3 cuối)
  arm C          target=[q,k,v,o,gate,up,down]_proj         layers=toàn bộ 30
dữ liệu / bài toán  klusai/ds-tf1-en-3m  tập con train=50.000 (seed 42)  epoch=2
                 conditional (system+prompt)->fable, loss chỉ trên token fable (mask -100), max_seq_len=512
tối ưu           AdamW  lr=2e-4  scheduler=cosine  warmup_ratio=0.03  bf16
                 per_device_batch=16  grad_accum=2  (hiệu dụng 32)  ~3.125 step/arm  phần cứng=Colab L4
đánh giá         held-out=500 dòng (validation, seed 42)
                 chính=perplexity (teacher-forced, có trọng số token)
                 reference-free=Distinct-1/2, Self-BLEU, Flesch  (100 bản sinh/arm; temp 0.8, top-p 0.9, rep-pen 1.3)
                 llm-judge=Qwen2.5-7B-Instruct 4-bit (bitsandbytes), greedy, rubric chỉ trả JSON, 4 trục 1-10
                 n=50 bản sinh/arm (50 dòng đầu của tập held-out), max_new_tokens=400, có seed
export           merge -> GGUF Q8_0 (llama.cpp) -> Ollama Modelfile -> tinystory-vn config/models.json
```

## Phụ lục B · Các bản ghi quyết định (ADR)

| ADR | Quyết định |
|---|---|
| **0001** | Model nhỏ đã pretrain + LoRA (không train từ đầu, không full fine-tune model lớn). |
| **0002** | Ablation vị trí đặt LoRA *chính là* đóng góp (arm A/B/C + base). |
| **0003** | Đánh giá đơn giản hoá một-judge thay cho panel 3-judge của bài báo (đã chạy: Qwen2.5-7B-Instruct, n = 50/arm). |
| **0004** | Bàn giao qua `tinystory-vn` (GGUF/Ollama), không app riêng. |
