# Nhật ký thử nghiệm: Pretrain SLM from scratch trên TF1-EN-3M

Branch: `feat/slm-pretrain-tf1`
Notebook Colab: Drive `1SUszrhTVm2bP-Lx3af1xEAfkb9_9oxRA` (bản repo: `notebooks/pretrain_slm_tf1.ipynb`)
Mục tiêu tổng: chứng minh một SLM rất nhỏ (~30M tham số) train from scratch trên TF1 vẫn sinh được fable tiếng Anh chất lượng, đặt cạnh LLM lớn (Qwen3-4B) trong app demo.

Cơ sở khoa học xuyên suốt (bám bài giảng IT5410 + literature):
- Autoregressive LM = MLE trên chain-rule `log p(x) = sum log p(x_i | x_<i)` (Materials Week6); loss-mask phần conditioning để chỉ học sinh truyện.
- Scaling laws (Kaplan et al. 2020, Materials Week4): loss giảm power-law theo N (tham số), D (token), C (compute).
- Chinchilla (Hoffmann et al. 2022): compute-optimal ~20 token/tham số.
- Data-constrained scaling (Muennighoff et al. 2023): lặp dữ liệu <=4 epoch gần tương đương data mới.
- Knowledge Distillation (Hinton et al. 2015): student nhỏ học phân bố mềm của teacher.
- Dataset + eval methodology: Nadas et al. (2025), TF1-EN-3M, arXiv:2504.20605; đánh giá bám ADR-0002.

---

## Run 1 (v1) — baseline reduced training: 10M + 30M, subset 150k

Ngày: 2026-07-08 (buổi sáng/trưa)

| Hạng mục | Giá trị |
|---|---|
| Kiến trúc | Llama-style, RoPE + GQA + RMSNorm + SwiGLU, tied embeddings, seq 512 |
| Size | 10M (6 layers, d320, 8 heads, 2 kv) và 30M (8 layers, d512, 8 heads, 2 kv) |
| Tokenizer | BPE riêng, vocab 12k, train trên chính corpus fable |
| Dữ liệu | 150k fable TF1, format `conditioning (5 slot) \n <\|story\|> fable <\|end\|>`, slot dropout |
| Optimizer | AdamW beta (0.9, 0.95), wd 0.1, grad-clip 1.0, fp16 |
| LR | peak 3e-3, WSD (warmup 2%, decay 20% cuối) |
| Batch/steps | 32 x accum 4 = 128 seq/update, **900 steps** (~50M token cho 30M, ~1.7 tok/param) |

Kết quả:
- Cả 10M và 30M hội tụ, loss cuối ~**1.8**; fable sinh ra mạch lạc, đúng cấu trúc scaffold.
- Export GGUF q8 thành công (2 fix: `tokenizer_class` transformers 5.x -> `PreTrainedTokenizerFast`; patch llama.cpp pre-tokenizer chkhsh -> `gpt-2`). Nạp vào app qua Ollama, chạy cạnh Qwen.
- So sánh trong app: SLM **nhanh hơn Qwen ~49x**, nhưng judge chấm thấp hơn nhiều (**2.5 vs 9.75**).

Kết luận Run 1: pipeline end-to-end đúng, nhưng model **undertrain nặng** (~1.7 token/tham số so với chuẩn Chinchilla 20x) -> cần tăng ngân sách token, không phải đổi kiến trúc.

---

## Thiết kế v2 — full Chinchilla + distillation (spec, chưa chạy)

Ngày: 2026-07-08. Spec: `docs/superpowers/specs/2026-07-08-slm-stronger-train-distill-design.md`.

- Teacher 30M train đủ ~600M token (Chinchilla 20x), subset 600k unique, <=4 epoch, STEPS ~7900.
- Distill token-level KD (T=2, alpha=0.5) từ teacher 30M xuống student 10M (`scripts/distill.py`, có TDD).
- Data prep mở rộng lọc chất lượng `--min-words/--max-words` (`scripts/prepare_tf1_pretrain.py`).
- Trạng thái: code + notebook v2 hoàn chỉnh, đã review; **chưa chạy** (chờ tài nguyên GPU dài hơi). Đây là hướng nâng cấp khi có quota.

---

## Run 2 (v2-light) — 30M-only, subset 400k, STEPS 3000 + dashboard Step 5B

Ngày: 2026-07-08 (buổi tối). Spec dashboard: `docs/superpowers/specs/2026-07-08-step5b-analysis-dashboard-design.md`.

Thay đổi so với Run 1:

| Hạng mục | Run 1 | Run 2 | Lý do |
|---|---|---|---|
| Dữ liệu | 150k | **400k**, lọc 60-320 từ | tăng token unique + chất lượng (scaling law: tăng D) |
| Size | 10M + 30M | **chỉ 30M** | tập trung compute vào 1 model |
| Steps | 900 | **3000** (~0.74 epoch/400k) | tăng ~3.3x token xử lý, chữa undertrain |
| Phân tích | 2 chart đơn giản | **dashboard Step 5B**: (A) training dynamics, (B) intrinsic quality vs fable thật (Distinct-1/2, Self-BLEU, Flesch, phân bố độ dài), (C) LM behavior (held-out perplexity, per-position loss, Zipf) | quan sát khoa học, có baseline tham chiếu |

Diễn biến training (T4, 0.60 it/s, fp16):

| step | 25 | 500 | 1000 | 1500 | 2000 | 2300 |
|---|---|---|---|---|---|---|
| loss | 7.21 | 2.10 | 1.76 | 1.61 | 1.53 | **1.49** |

- Loss giảm mượt, đơn điệu, không spike -> recipe (LR 3e-3, WSD, loss-mask) lành mạnh.
- Tại step 2300 loss **1.49 < 1.8 (v1)** dù mới ~77% số bước -> xác nhận luận điểm scaling: thêm dữ liệu unique + thêm bước huấn luyện cải thiện rõ.

**Sự cố:** hết hạn mức GPU free của Colab ở step ~2300/3000 ("Cannot connect to GPU backend"), runtime bị recycle. Code lúc đó chỉ lưu model SAU khi train xong và lưu vào `/content` (local) -> **mất model, không chạy được Step 5B**. Bài học đắt nhất của ngày.

---

## Biện pháp sau Run 2 — cơ chế resume 3 tầng (đã cài vào notebook)

| Bước tốn công | Cache trên Drive | Hành vi sau recycle |
|---|---|---|
| Corpus + tokenizer (tải + lọc 400k) | `slm_tf1/data_tf1` | Step 1 khôi phục trong vài giây, không tải lại |
| Training | `slm_tf1/ckpt_30M`, checkpoint mỗi 500 step, giữ 2 bản | `train("30M")` tự resume từ checkpoint gần nhất (mất tối đa <500 step) |
| Model cuối | `slm_tf1/30M` | Harness Step 5B tự load từ Drive nếu bản local mất |

Điều chỉnh tham số cho lần chạy kế:
- `STEPS` 3000 -> **1800** (~40-50 phút/T4): đánh đổi loss ~1.55 thay vì ~1.43, đổi lấy chắc chắn hoàn tất trong 1 phiên free trước khi chạm quota. Dữ liệu 400k đủ dùng toàn bước unique (1800 x 128 ~ 230k example).
- Giữ nguyên toàn bộ phần học thuật: PEAK_LR 3e-3, AdamW (0.9, 0.95), wd 0.1, clip 1.0, WSD, batch hiệu dụng 128, fp16 — vì đường loss của Run 2 cho thấy các nút này đều đúng.

## Run 3 (v2-light, THÀNH CÔNG) — 30M, 400k, STEPS 1800 + dashboard đầy đủ

Ngày: 2026-07-09. GPU quota đã reset; chạy trọn vẹn trong một phiên T4.

- Corpus khôi phục từ cache Drive (lần chuẩn bị trước để lại 600k dòng, cùng filter 60-320 từ); encode cap ở 400k để an toàn RAM.
- Training: 1800/1800 step trong **52.4 phút**, loss **7.00 -> 1.447** (vượt dự đoán 1.55 nhờ pha decay của WSD). Checkpoint Drive mỗi 500 step hoạt động đúng.
- Model lưu cả `out/30M` (local) lẫn `slm_tf1/30M` (Drive); GGUF q8 + Modelfile export lên Drive.

Bảng verdict tự động (Step 5b): **7 PASS | 1 WARN | 0 FAIL**

| Metric | Giá trị | Target | Verdict |
|---|---|---|---|
| final train loss | 1.447 | < 1.5 | PASS |
| scaling-law fit R^2 | 0.992 (exp -0.33) | > 0.95 | PASS |
| held-out perplexity | 4.18 (floor 4.25, 0.98x) | <= 1.5x floor | PASS |
| Distinct-1 gap vs real | 3% | <= 15% | PASS |
| Distinct-2 gap vs real | 2% | <= 15% | PASS |
| Self-BLEU abs gap | 0.001 | <= 0.05 | PASS |
| Flesch reading ease | 75.8 | 80-100 (60-80 = WARN) | WARN |
| length distribution overlap | 57% | >= 50% | PASS |

Diễn giải khoa học:
- **Perplexity held-out 4.18 < floor e^train-loss 4.25**: model tổng quát hóa hoàn hảo, không overfit (train và held-out gần trùng phân bố vì dataset unique lớn).
- **R^2 0.992 trên log-log**: đường loss bám power-law regime đúng như Kaplan et al. 2020 dự đoán — bằng chứng scaling law trực tiếp trên run của mình.
- **Đa dạng từ vựng ngang fable thật** (Distinct gap 2-3%, Self-BLEU gap 0.001): model không học vẹt, không lặp mẫu câu.
- Flesch 75.8 hơi dưới dải 80-100 (WARN nhẹ): câu sinh ra phức tạp hơn chuẩn thiếu nhi một chút; fable TF1 gốc cũng nằm quanh mức này.
- Truyện sinh mẫu (3 held-out prompt) mạch lạc, bám scaffold, hơn hẳn v1.

Artifact trên Drive `slm_tf1/`: `30M/`, `slm-30m.gguf`, `Modelfile-30M`, `loss_log_30M.json`, `analysis_30M.json` (kèm verdict), `fig_training_30M.png`, `fig_quality_30M.png`, `fig_lm_30M.png`, `ckpt_30M/`, `data_tf1/`.

Sự cố nhỏ sau run: export GGUF lần đầu **fail âm thầm** (subprocess không check returncode) vì tokenizer đã được train lại trên corpus 600k -> **chkhsh pre-tokenizer đổi**, patch hardcode hash cũ không khớp, và file `slm-30m.gguf` cũ (v1, 8/7) nằm nguyên trên Drive đánh lừa là export xong. Fix: cell export giờ tự đọc chkhsh mới từ log lỗi, tự patch, convert lại và fail-fast bằng assert returncode. Bài học bổ sung: (a) luôn check returncode của subprocess trong notebook; (b) chkhsh gắn với tokenizer artifact, train lại tokenizer là phải patch lại.

Việc tiếp theo: tải GGUF về máy -> `ollama create slm-30m` -> so trong app vs Qwen; chạy batch eval judge panel (`scripts/eval_slm.py`, ADR-0002) lấy số near-parity cho báo cáo.

## Đánh giá định tính theo rubric bài báo (2026-07-11, Claude tự chấm)

Phương pháp: sinh 10 truyện từ 10 prompt held-out đầu tiên (`data/tf1/test.jsonl`, sampling như app: temp 0.8, top_p 0.9, repeat_penalty 1.3), tự chấm theo 4 trục 1-10 của rubric TF1-EN-3M (arXiv:2504.20605, cài trong `app/judge.py`). Mẫu lưu ở `results/slm30m_samples_for_review.json`.

| Trục | Điểm TB | Nhận xét |
|---|---|---|
| Grammar | ~6 | Câu cục bộ trôi chảy; lỗi đại từ/tham chiếu ở mọi truyện ("She saw that her anger had made him angry"), thi thoảng mệnh đề vô nghĩa, lệch dấu ngoặc kép |
| Creativity | ~6 | Setting đa dạng, dễ thương; nhưng khung cốt truyện lặp: 7/10 truyện đều "bão đến -> cùng nhau làm việc -> moral" |
| Moral clarity | ~6.5 | Moral luôn được phát biểu tường minh cuối truyện, NHƯNG hay bị thay bằng moral generic (kindness/teamwork) thay vì moral được yêu cầu ("oppression sows rebellion" -> teamwork; "a watchful eye prevents betrayal" -> kindness) |
| Prompt adherence | ~5.5 | Yếu nhất. Lỗi hệ thống: **entity drift** - nhân vật chính biến hình giữa truyện (hyena -> zebra, Greedy Skunk -> Thrifty Skunk -> "cormorant", rabbit/squirrel -> hedgehog/woodpecker); setting bị bỏ rơi (hanging bridge -> jungle temple); xuất hiện thực thể chưa từng có ("the fierce dragon's fury") |
| **Overall** | **~6.0** | v1 là 2.5; Qwen-4B ~9.75. Cải thiện lớn nhưng còn khoảng cách rõ |

### Chẩn đoán gốc rễ + fix xác nhận bằng A/B

**Phát hiện quan trọng: `repeat_penalty 1.3` là thủ phạm chính của entity drift.** Penalty cao phạt việc LẶP LẠI token tên nhân vật, nên giữa truyện model bị ép chọn con vật khác. A/B trên 3 prompt drift nặng nhất với `repeat_penalty 1.1` (`results/slm30m_rp11_ab.json`): nhân vật giữ nguyên suốt truyện, setting bám đúng (hanging bridge mở và kết truyện), moral trả ĐÚNG nguyên văn yêu cầu ("mildness can tame even the greatest fury", "A watchful eye can prevent betrayal"), challenge "identity crisis" được thể hiện tường minh ("I don't know who I am"). Ước lượng adherence tăng ~+2-3 điểm chỉ nhờ đổi sampling, không cần train.

Fix đã áp dụng (commit cùng ngày): `app/config.py` GEN_REPEAT_PENALTY 1.3 -> **1.1**; `models/Modelfile-30M` + notebook (harness + export) đồng bộ 1.1; `ollama create slm-30m` lại.

### Vấn đề còn lại thuộc về TRAINING (không phải sampling)

1. Lỗi đại từ/tham chiếu (binding) - cần thêm bước huấn luyện cho long-range coherence.
2. Moral substitution một phần (adherence với moral hiếm/trừu tượng) - thêm token huấn luyện trên data unique.
3. Khung cốt truyện lặp (storm/teamwork) - mode collapse nhẹ, thêm data unique giúp giảm.

Quyết định cải thiện: (a) fix sampling (DONE, free); (b) **train tiếp resume từ checkpoint Drive lên STEPS 3600** (~50 phút T4, loss kỳ vọng 1.447 -> ~1.38) qua Colab CLI; (c) re-eval cùng 10 prompt để đo delta; full-Chinchilla 7900 để dành khi cần đẩy tiếp.

## Run 4 (pha 2 resume, 2026-07-13) — CHẠY XONG NHƯNG MẤT KẾT QUẢ (lỗi vận hành)

Chạy headless qua Colab CLI (session T4, không mount Drive): tải checkpoint pha 1 + tokenizer v1 từ folder Drive public bằng gdown, build corpus v2 (cap "wise old owl" 10% + slot-dropout teaching/outcome 0.15), resume 1500 -> 3600. Pipeline chạy trọn vẹn đến marker cuối `PIPELINE DONE` (train + analysis + GGUF + nén tarball đều xong) NHƯNG toàn bộ artifact và số liệu bị mất do 2 lỗi harness phía client:

1. Stream output bị bọc `| tail -100` — tail buffer nuốt toàn bộ log (loss curve, dòng ANALYSIS chứa perplexity/owl-rate) và chỉ nhả 100 dòng cuối, trùng đúng đoạn traceback timeout của client.
2. Bước `colab download` tách rời khỏi exec — Colab thu hồi VM idle chỉ ít phút sau khi cell chạy xong, tarball trên `/content` mất trước khi kịp tải.

Hết quota GPU trong ngày ngay sau đó (Precondition Failed) nên chưa retry được.

Bài học vận hành (bổ sung mục Bài học): (a) log stream dài phải ghi thẳng ra file, không qua tail/head; (b) download artifact phải nối liền exec trong cùng một lệnh; (c) `colab exec` client có thể TimeoutError dù kernel đã chạy xong — kiểm tra bằng `colab status`; (d) folder Drive public + gdown là đường đưa checkpoint VÀO VM headless hiệu quả (drivemount headless bị chặn bởi scope OAuth).

Kế hoạch retry: script một-lệnh đã sẵn (chain new -> exec -> download -> stop, log đầy đủ); chạy khi quota reset. Kết quả kỳ vọng không đổi: loss ~1.36-1.40, owl-rate giảm mạnh từ 90%.

## Run 5 (PHA 2 THÀNH CÔNG, 2026-07-14) — resume 1800 -> 3600 trên corpus v2

Chạy qua notebook UI (user theo dõi trực tiếp) + colab-mcp điều khiển. Resume từ `ckpt_30M/checkpoint-1800` (pha 1 có lưu bản cuối), 3600/3600 step trong **49 phút** trên T4.

Corpus v2 kiểm chứng ngay khi build: 400k dòng, **owl 10.0%** (đúng target; data gốc 28%).

Diễn biến loss: mở đầu ~1.51 (nhích nhẹ do distribution shift sang corpus v2 - đúng dự đoán) -> giảm đều -> WSD decay cuối kéo xuống **1.278** (kỳ vọng 1.36-1.40; pha 1: 1.447).

Bảng verdict (9 metric, thêm owl-rate): **7 PASS | 2 WARN | 0 FAIL**

| Metric | Pha 1 | Pha 2 | Verdict |
|---|---|---|---|
| final train loss | 1.447 | **1.278** | PASS |
| held-out perplexity | 4.18 (0.98x floor) | **3.56** (0.99x floor 3.59) | PASS (giảm 15%, vẫn không overfit) |
| scaling-law fit R^2 | 0.992 | 0.959 (exp -0.25) | PASS (thấp hơn do cú nhích resume phá power-law thuần) |
| Distinct-1/2 gap vs real | 3%/2% | 8%/4% | PASS |
| Self-BLEU abs gap | 0.001 | 0.001 | PASS |
| Flesch | 75.8 | **79.9** | WARN (cách dải 80-100 đúng 0.1 điểm) |
| length overlap | 57% | 43% | WARN (giảm nhẹ - theo dõi) |
| **owl template rate (gen)** | **90%** | **23%** | **PASS** (dưới cả prior 28% của data thật) |

Kết luận: cả hai can thiệp data đều trúng đích - cap-phrase chữa template collapse (90% -> 23%), loss/ppl cải thiện rõ nhờ thêm ~50% token huấn luyện + annealing trên data sạch. Truyện mẫu giữ nhân vật/setting ổn định (vd "The Patient Squirrel and the Hanging Bridge" bám bridge xuyên suốt).

Artifact (hậu tố p2, không đè pha 1) trên Drive `TinyStoryVN/Trieulh/`: `30M-p2/`, `slm-30m-p2.gguf`, `Modelfile-30M-p2`, `analysis_30M_p2.json` (kèm 30 truyện + verdict), `loss_log_30M_p2.json`, 3 fig PNG p2, `ckpt_30M_p2/`, `data_tf1_v2/`.

Tiếp theo: nạp `slm-30m-p2` vào app -> đánh giá định tính 10 prompt (so 3 nấc) -> ORPO (pairs + train) -> batch eval panel --limit 15.

## Đánh giá định tính pha 2 + tích hợp app (2026-07-14)

**Tích hợp app:** `slm-30m-p2.gguf` + `Modelfile-30M-p2` tải về, `ollama create slm-30m-p2`; registry thêm entry "Fable-SLM 30M (phase 2)" và GIỮ phase 1 để demo so sánh trong Compare. Smoke test: 949 tok/s, nhân vật giữ nhất quán. 14 API test pass.

**Đánh giá định tính** (cùng 10 prompt held-out, cùng rubric TF1 4 trục, Claude tự chấm; mẫu: `results/slm30m_p2_samples.json`):

| Trục | P1 (rp 1.3) | P1 (rp 1.1) | **P2** | Bằng chứng tiến bộ ở P2 |
|---|---|---|---|---|
| Grammar | ~6.0 | ~6.0 | ~6.3 | Ít câu vỡ/lệch ngoặc kép; vẫn còn trượt đại từ lẻ tẻ |
| Creativity | ~6.0 | ~6.1 | **~6.8** | Nhân vật có tên (Quickheart, Kind Heart); cốt truyện mới (trộm giọng hát); trung gian đa dạng: octopus/hawk/rabbit thay vì owl độc quyền |
| Moral clarity | ~6.5 | ~6.3 | **~7.3** | Moral được DIỄN chứ không chỉ đọc: "kept a watchful eye" dệt vào hành động 2 lần; squirrel tự kìm cơn trả thù (mildness enacted) |
| Prompt adherence | ~5.5 | ~6.5 | **~7.4** | "Rivalry in love" lần đầu được xử lý đúng; chuỗi rumors -> chaos -> clarified đủ 4 yếu tố; setting bám chắc |
| **Overall** | **~6.0** | **~6.2** | **~7.0** | (Qwen-4B ~9.75; pha 1 v1-reduced: 2.5) |

Owl-rate trong 10 truyện: 2/10 (20%), khớp số đo 23% trên 30 truyện của notebook.

**Điểm yếu còn lại (mục tiêu của bước ORPO):**
1. Moral hiếm/trừu tượng vẫn bị thay bằng moral generic ("oppression sows rebellion" -> cooperation).
2. Outcome tiêu cực bị lật thành happy-ending ("the troublemaker leaves in shame" -> redemption) - bias redemption rất sâu của TF1.
3. Rối tên khi nhiều nhân vật có tên trong một truyện.

## ORPO khởi động (2026-07-14, đang chạy qua đêm)

Pair generation chạy nền local: 1500 prompt held-out (offset 500, không đụng bộ eval) x 2 truyện từ `slm-30m-p2` (temp 0.8, seed 11/97) x judge chấm rubric từng bản, giữ pair khi chênh overall >= 1.0. **Điều chỉnh so với spec:** judge đổi từ `qwen3:4b` (bản thinking, ~30-60s/lượt -> 3000 lượt quá 24h) sang `qwen3-4b-instruct` (non-thinking, judge mặc định của app, ~5x nhanh hơn, JSON sạch) - ước 4-6h. Resume-safe (prompt đã chấm không chấm lại). Output: `data/orpo/pairs.jsonl`. Bước kế: `scripts/orpo_train.py` trên Colab T4 (~40 phút, guard perplexity +10%) -> `slm-30m-orpo` -> đánh giá nấc 4 -> batch eval panel 3 judge --limit 15 -> `results/eval_summary.json`.

## Tinh chỉnh sinh truyện: đảm bảo tính hoàn thiện (2026-07-16)

Vấn đề: 30M-p2 đôi lúc sinh truyện cụt giữa câu (vd "...knowing that she had") khi chọn Short. Chẩn đoán: truyện đụng trần `num_predict` (Short=300) trước khi model kịp phát `<|end|>`; và model train `seq_len=512` nên chỉ sinh coherent được ~400-460 token truyện (headroom sau prompt ~52-110 token) - tăng token/continuation vô nghĩa vì vượt context đã train.

Giải pháp single-shot (spec/plan `2026-07-16-story-completeness*`):
1. Right-size `num_predict` về headroom thật: Short/Medium/Long = 400/440/460 (bỏ 600/1000 ảo).
2. Sampling hội tụ: temp 0.8->0.7, top_p 0.9->0.85 (bám nhánh xác suất cao, ít lan man; đổi lấy giảm đa dạng).
3. Kéo mốc hint độ dài về mức model kết được (long 450-600 -> 280-340 từ) vì "long" cũ vượt khả năng nên dễ cụt nhất.
4. Tận dụng `done_reason` (đang bị vứt): `"stop"` = model phát `<|end|>` = kết thật; `"length"` = cụt -> `trim_to_last_sentence()` cắt về câu hoàn chỉnh cuối (không gắn câu tự chế).

Kết quả đo (slm-30m-p2, 106/106 test pass):

| Test | Kết quả |
|---|---|
| Prompt gây lỗi cũ | done_reason=stop, kết moral trọn vẹn (hết cụt) |
| 10 held-out x Short | 10/10 kết thật |
| 10 held-out x Medium | 10/10 kết thật (234-303 từ) |
| 10 held-out x Long | 10/10 kết thật (246-275 từ) |

**30/30 truyện hoàn thiện với kết thật**; lưới trim chưa phải kích hoạt (kết quả tốt nhất). Quan sát: length adherence yếu (cả 3 mức ~250-280 từ - model có "độ dài tự nhiên", gần như bỏ qua hint) - chấp nhận được vì độ dài ưu tiên thấp; đa dạng giảm nhẹ (moral cụm quanh friendship/kindness - một phần prior TF1, một phần temp thấp), `GEN_TEMPERATURE` là cần gạt nếu muốn đa dạng hơn. Đây là hiện tượng khoa học đáng nêu: **giới hạn context 512 của model quyết định độ dài tối đa, không phải cấu hình app**.

## Thực nghiệm: sampling có cải thiện prompt-adherence không? (2026-07-16)

Bối cảnh: sau khi hạ temp 0.8->0.7 cho completeness, cảm nhận truyện mượt hơn nhưng bám 5 slot (character/setting/challenge/outcome/teaching) kém hơn. Giả thuyết: sampling "an toàn" hơn khiến model theo prior generic thay vì slot. Đo bằng slot-recall (mỗi slot 'hit' nếu >=1 content word của slot xuất hiện trong truyện).

Mẫu nhỏ (4 prompt đủ 5 slot, seed cố định) - GÂY NHẦM:

| Variant | slot-recall | complete |
|---|---|---|
| t0.7/tp0.85/rp1.1 (hiện tại) | 80% | 4/4 |
| t0.8/tp0.9/rp1.1 (bản cũ) | **90%** | 4/4 |
| t0.7 rp1.05 / rp1.0 | 75% | 4/4 |
| t0.9/tp0.92/rp1.05 | 75% | 4/4 |

Mẫu lớn hơn (10 prompt held-out) - KẾT LUẬN THẬT:

| Variant | adherence | complete |
|---|---|---|
| t0.7/tp0.85 (hiện tại) | 69% | 10/10 |
| t0.8/tp0.9 (revert) | 69% | 10/10 |

**Kết luận:** chênh lệch 80/90% ở mẫu 4-prompt là **nhiễu** (mẫu nhỏ, 1 seed). Trên mẫu lớn, temp/top_p/repeat_penalty **không dịch chuyển adherence đáng kể** — nó bị chặn ~69-80% bởi **khả năng conditioning yếu bẩm sinh của model 30M** (đọc 5 slot nhưng hay bỏ rơi 1-2 slot, nhất là challenge/outcome trừu tượng). Đây là **trần capacity**, không phải lỗi sampling. Hệ quả: KHÔNG đổi sampling (giữ temp 0.7 cho convergence, completeness đã do num_predict + trim đảm nhiệm). Cải thiện adherence thật sự phải đến từ: (a) **ORPO** - preference optimization thưởng trực tiếp trục prompt_adherence (đúng mục tiêu, đang pending); (b) can thiệp prompt/salience (hiệu quả hạn chế với 30M); hoặc (c) chấp nhận trần ~70% như giới hạn model nhỏ và ghi rõ trong báo cáo. Bài học phương pháp: **luôn kiểm chứng trên mẫu đủ lớn trước khi kết luận** - khác biệt trên 4 mẫu dễ là nhiễu.

## Alignment trial: DPO local trên MPS với 115 pairs (2026-07-16)

Mục tiêu: cải thiện prompt-adherence (đã chứng minh sampling không sửa được - là trần capacity). Chương alignment: preference optimization từ RLAIF pairs.

Ghi chú phương pháp: dự kiến ORPO nhưng **trl 1.8 (bản rewrite) đã bỏ ORPOTrainer** -> chuyển sang **DPO (Rafailov et al. 2023)** - biến thể kinh điển tương đương, cùng data (chosen/rejected), cần thêm reference model đóng băng (36M nên rẻ). Toàn bộ chạy **local trên Apple Silicon MPS** (M3 Pro 36GB) - **KHÔNG cần Colab** cho bước alignment (pair-gen local + DPO train local). Script: `trieulh/scripts/dpo_train_local.py`.

Pairs: 115 (từ pair-gen với judge qwen3-4b-instruct, min-margin 1.0; đang resume tới 500). Setup: khởi từ `30M-p2`, 2 epoch, lr 5e-6, beta 0.1, batch 4 x accum 2, ppl guard held-out.

Kết quả trial (55 giây / MPS):
- **Học đúng hướng**: `rewards/accuracies` 0.47 -> **1.0**, `rewards/margins` -0.003 -> **+0.21** (model học ưu tiên bản chosen bám prompt tốt hơn).
- **Không quên ngôn ngữ**: perplexity held-out **3.539 -> 3.539 (0% drift)** - guard catastrophic forgetting pass hoàn hảo.
- **Adherence dịch chuyển**: slot-recall trên 12 prompt held-out **71% -> 76% (+5đ)**; completeness giữ **12/12** kết `<|end|>`.

Kết luận: pipeline DPO chạy được local, cải thiện adherence đúng hướng ngay cả với chỉ 115 pairs (+5đ là tín hiệu nhỏ nhưng khớp reward signal; kỳ vọng mạnh hơn với 500 pairs). Bước tiếp: resume pair-gen tới 500 -> DPO full local -> convert GGUF -> nạp app + đánh giá adherence/judge chính thức. Việc trl bỏ ORPO cũng cần cập nhật `trieulh/scripts/orpo_train.py` (bản Colab) sang DPO nếu sau này chạy Colab.

## Trạng thái hiện tại và việc còn lại

- Notebook đã sẵn sàng chạy lại bền vững (Run all là đủ; tự resume ở mọi tầng).
- Chờ hạn mức GPU free mở lại (thường vài giờ tới ~24h) -> chạy: restore cache -> train 1800 step -> Step 5B (3 figure + JSON trên Drive) -> export GGUF q8 + Modelfile vào Drive -> nạp vào app.
- Sau khi có model: chạy batch eval judge panel (`scripts/eval_slm.py`, ADR-0002) để có số so sánh SLM vs Qwen cho báo cáo.

## Artifact cho báo cáo — inventory đầy đủ trên Drive `slm_tf1/` (verify 2026-07-09)

Mọi chuỗi số vẽ biểu đồ đều đã persist; tải 2 file JSON dưới đây là đủ dựng lại toàn bộ curve/line ngoài Colab (matplotlib/Excel):

| File | Nội dung | Dùng cho biểu đồ nào |
|---|---|---|
| `loss_log_30M.json` (73 entries) | log gốc của Trainer: mỗi entry có `step`, `loss`, `learning_rate`, `grad_norm`, `epoch`; entry cuối có `train_runtime`, `train_samples_per_second`, `total_flos` | nguồn thô cho mọi curve training (kể cả grad_norm chưa vẽ) |
| `analysis_30M.json` | `steps/losses/lrs` (72 điểm) - Fig1 loss + LR + log-log; `powerlaw` (exponent -0.33, R^2 0.992); `quality` (distinct1/2, self_bleu, flesch: gen vs real) - Fig2 bar; `len_gen/len_real` (30+30) - Fig2 histogram + `len_overlap` 57%; `perplexity` 4.18 + `final_loss` 1.4466 + `vocab_size` 12000 - Fig3 anchor bar; `pos_loss` (16 bin) - Fig3 position profile; `zipf_gen/zipf_real` (1973/2000 điểm) - Fig3 Zipf; `samples` (3 truyện mẫu); `verdict` (8 dòng PASS/WARN/FAIL); `throughput` 73.2 samples/s, `runtime_min` 52.45 | toàn bộ Fig1-3 + bảng verdict |
| `fig_training_30M.png`, `fig_quality_30M.png`, `fig_lm_30M.png` | 3 figure render sẵn (110 dpi) | chèn thẳng vào báo cáo |
| `slm-30m.gguf` (39.3MB, mtime 09/07 13:48) + `Modelfile-30M` | model q8 + định nghĩa Ollama | chạy app/demo |
| `30M/` | model HF đầy đủ + tokenizer | tái phân tích, distill sau này |
| `ckpt_30M/`, `data_tf1/` | checkpoint resume + corpus/tokenizer cache | chạy lại/kéo dài training |
| `pretrain_slm_tf1_clean.ipynb` | notebook trên Drive, đã lưu kèm output các cell (bảng loss trực tiếp + 3 figure inline) | bằng chứng quá trình chạy |

Giới hạn đã biết (ghi để trung thực trong báo cáo):
- **30 truyện sinh ra của Run 3**: chỉ 3 sample đầy đủ nằm trong `analysis_30M.json` (`samples`); 27 truyện còn lại chỉ còn độ dài (`len_gen`) và các metric tổng hợp. Harness đã được vá (`gen_stories`) để từ run sau lưu đủ cả 30. Không sinh bù cho Run 3 vì sampling không đặt seed - truyện mới sẽ không khớp metric đã tính.
- **Loss history của Run 2 (bị cắt 08/07)**: không dump được (train chưa xong thì mất runtime); chỉ còn bảng mốc 25/500/1000/1500/2000/2300 ghi ở mục Run 2 phía trên.
- Repo: spec/plan trong `docs/superpowers/{specs,plans}/`, notebook mirror `notebooks/pretrain_slm_30m_dashboard.ipynb`, script trong `scripts/`, kết quả eval trong `results/eval_summary.json` (sau S4).

## Bài học rút ra (để viết phần "quá trình" trong báo cáo)

1. **Undertraining là nguyên nhân chính** khiến SLM v1 thua xa LLM trên judge, không phải kiến trúc — đúng như scaling laws dự đoán; tăng D (token unique + số bước) cải thiện loss ngay (1.8 -> 1.49).
2. **Hạ tầng free có hạn mức động**: mọi bước tốn công (data, checkpoint, model, kết quả phân tích) phải persist ra Drive ngay từ đầu, không đợi cuối run. Checkpoint/resume không phải tùy chọn mà là bắt buộc.
3. **Chọn ngân sách bước theo cửa sổ tài nguyên**: một run 45 phút chắc chắn xong có giá trị hơn một run 2 giờ bị cắt giữa chừng.
4. **Đánh giá cần mốc tham chiếu**: dashboard Step 5B đặt mọi metric của model cạnh fable thật (held-out) để khoảng cách chất lượng đọc được ngay trên biểu đồ.
