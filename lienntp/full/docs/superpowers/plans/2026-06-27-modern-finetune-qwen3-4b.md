# Modern Fine-tune Qwen3-4B (SFT DoRA+NEFTune → ORPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nâng chất lượng truyện bằng base Qwen3-4B + pipeline 2 tầng hiện đại (SFT QLoRA+DoRA+NEFTune nhẹ tay → ORPO preference alignment), dùng dữ liệu hiện có, chạy Colab T4; đổi web app sang 2 model 4B (base + fine-tuned), bỏ 1.7B.

**Architecture:** Phần code repo (app config, Modelfile) đổi nhỏ và test được local. Phần fine-tune là một notebook Colab gồm: SFT (DoRA+NEFTune+responses-only) → sinh preference dataset (chosen=truyện thật, rejected=output base-4B) → ORPO → merge → GGUF. Chạy notebook tương tác qua colab-mcp trên T4.

**Tech Stack:** Python, FastAPI, pytest; Unsloth + TRL (SFTTrainer/ORPOTrainer) + DoRA + NEFTune; Qwen3-4B (`unsloth/Qwen3-4B-Instruct-2507`); Ollama (GGUF q8); colab-mcp.

## Global Constraints

- Base model fine-tune: `unsloth/Qwen3-4B-Instruct-2507`. App base (Ollama): `qwen3:4b`. App tuned (Ollama): `fable-tuned`.
- App SAU khi xong: chỉ 2 model 4B — `BASE_MODEL=qwen3:4b`, `TUNED_MODEL=fable-tuned`. KHÔNG còn 1.7B trong app.
- Fine-tune nhẹ tay: SFT LR 5e-5, 1–2 epoch, `use_dora=True`, `neftune_noise_alpha=5`, train-on-responses-only, `enable_thinking=False`, seq 2048, batch 1 + grad accum 8.
- ORPO: reference-free, LR 8e-6, beta 0.1, 1 epoch, batch 1 + grad accum, seq ≤2048 (hạ 1024 nếu OOM).
- Preference data KHÔNG dùng teacher ngoài: chosen=truyện thật trong dataset, rejected=output base-4B tự sinh.
- Mọi chuỗi hiển thị tiếng Việt có dấu. Test backend phải tiếp tục pass.
- Đã biết (từ lần trước): trên Colab phải `%pip uninstall -y torchao` sau khi cài unsloth để tránh lỗi `_wrap_tensor_autograd`; generation cần `repetition_penalty` để tránh lặp.

---

## File Structure

```
app/config.py            # BASE_MODEL default -> qwen3:4b
app/main.py              # MODEL_INFO mô tả (vẫn 2 lựa chọn base/tuned, đều 4B)
tests/test_api.py        # test default base = qwen3:4b
notebooks/finetune_qwen3_4b_orpo.ipynb   # notebook mới (SFT+preference+ORPO+export)
ollama/Modelfile         # FROM gguf 4B mới
```

---

## Task 1: App — đổi base model mặc định sang Qwen3-4B

**Files:**
- Modify: `app/config.py`
- Modify: `app/main.py` (MODEL_INFO desc)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `app.config.BASE_MODEL`, `TUNED_MODEL`; endpoint `GET /models` (Task 14 cũ).
- Produces: `BASE_MODEL` mặc định `"qwen3:4b"`.

- [ ] **Step 1: Viết/đổi test (TDD)** trong `tests/test_api.py` — thêm:

```python
def test_default_base_model_is_qwen3_4b():
    from app import config
    import importlib
    # đảm bảo không bị env override khi chạy test
    import os
    assert os.getenv("FABLE_BASE_MODEL") in (None, "qwen3:4b")
    assert config.BASE_MODEL == "qwen3:4b"
    assert config.TUNED_MODEL == "fable-tuned"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_api.py::test_default_base_model_is_qwen3_4b -v`
Expected: FAIL (`BASE_MODEL == "qwen3:1.7b"` hiện tại)

- [ ] **Step 3: Sửa `app/config.py`**

Đổi dòng:
```python
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "qwen3:1.7b")
```
thành:
```python
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "qwen3:4b")
```

- [ ] **Step 4: Cập nhật mô tả trong `app/main.py` MODEL_INFO** (nếu có nhắc 1.7B thì bỏ; mô tả chung):

```python
MODEL_INFO = {
    "base": {
        "label": "Mô hình nền (chưa train)",
        "desc": "Qwen3-4B gốc, chưa fine-tune trên truyện ngụ ngôn — mốc so sánh “trước khi train”.",
    },
    "tuned": {
        "label": "Mô hình đã fine-tune",
        "desc": "Qwen3-4B đã fine-tune (SFT + ORPO) trên truyện ngụ ngôn — kết quả “sau khi train”.",
    },
}
```

- [ ] **Step 5: Chạy test để xác nhận PASS + full suite**

Run: `pytest -q`
Expected: PASS toàn bộ.

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/main.py tests/test_api.py
git commit -m "feat: switch app base model to Qwen3-4B (drop 1.7B)"
```

---

## Task 2: Notebook fine-tune Qwen3-4B (SFT DoRA+NEFTune → preference → ORPO → export)

> Không unit-test được. Deliverable = file notebook hợp lệ (nbformat 4) với đủ các cell. Kiểm chứng chạy thật ở Task 4 (Colab).

**Files:**
- Create: `notebooks/finetune_qwen3_4b_orpo.ipynb`

**Interfaces:**
- Consumes: `data/processed/{train,val}.jsonl` (đưa vào Colab ở Task 4).
- Produces: model merge 4B → GGUF q8 (`*.Q8_0.gguf`) tải về máy.

**Cách tạo notebook:** viết script sinh `.ipynb` (tránh lỗi escape JSON) rồi chạy. Mỗi cell dưới đây là một code cell (trừ cell mô tả là markdown). Nội dung cell PHẢI đúng như sau:

- [ ] **Step 1: Viết script generator** `scratch/gen_nb_4b.py` (đặt trong thư mục scratchpad, KHÔNG commit) tạo notebook với các cell:

**Cell 0 (md):**
```
# Fine-tune Qwen3-4B: SFT (DoRA+NEFTune) -> ORPO — truyện ngụ ngôn tiếng Việt
Chạy trên Colab T4. Thứ tự: cài đặt -> hyperparams -> nạp data -> nạp model -> SFT -> sinh preference -> ORPO -> sinh thử -> export GGUF.
```

**Cell 1 (install) — kèm fix torchao đã biết:**
```python
%pip install -q unsloth
%pip uninstall -q -y torchao
print("cài đặt xong (đã gỡ torchao tránh xung đột)")
```

**Cell 2 (hyperparams):**
```python
MODEL_NAME      = "unsloth/Qwen3-4B-Instruct-2507"
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True
# LoRA + DoRA
LORA_R          = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.0
USE_DORA        = True
# SFT
SFT_LR          = 5e-5
SFT_EPOCHS      = 2
NEFTUNE_ALPHA   = 5
BATCH_SIZE      = 1
GRAD_ACCUM      = 8
WARMUP_STEPS    = 5
SEED            = 42
# ORPO
ORPO_LR         = 8e-6
ORPO_BETA       = 0.1
ORPO_EPOCHS     = 1
ORPO_MAX_LEN    = 2048      # hạ 1024 nếu OOM
ORPO_PROMPT_LEN = 1024
# Data / format
TRAIN_PATH      = "train.jsonl"
VAL_PATH        = "val.jsonl"
SYSTEM_PROMPT   = "Bạn là người kể truyện ngụ ngôn cho trẻ em."
ENABLE_THINKING = False
MERGED_DIR      = "qwen3-4b-fable-merged"
print("hyperparams:", MODEL_NAME, "| SFT", SFT_LR, SFT_EPOCHS, "| DoRA", USE_DORA, "| ORPO", ORPO_LR)
```

**Cell 3 (nạp model + LoRA/DoRA):**
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED, use_dora=USE_DORA)
print("model loaded, DoRA =", USE_DORA)
```

**Cell 4 (data SFT format):**
```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": TRAIN_PATH, "val": VAL_PATH})
def to_text(ex):
    msgs = [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":ex["instruction"]},
            {"role":"assistant","content":ex["output"]}]
    try: t = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=ENABLE_THINKING)
    except TypeError: t = tokenizer.apply_chat_template(msgs, tokenize=False)
    return {"text": t}
sft_ds = ds.map(to_text)
print(sft_ds)
```

**Cell 5 (SFT train: NEFTune + responses-only):**
```python
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=sft_ds["train"], eval_dataset=sft_ds["val"],
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS, num_train_epochs=SFT_EPOCHS, learning_rate=SFT_LR,
        logging_steps=5, eval_strategy="epoch", output_dir="sft_out",
        dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH, seed=SEED,
        neftune_noise_alpha=NEFTUNE_ALPHA),
)
trainer = train_on_responses_only(
    trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
trainer.train()
```

**Cell 6 (sinh preference dataset: chosen=thật, rejected=base):**
```python
import json
# rejected = sinh bằng model NỀN (tắt adapter) để có "output base"
FastLanguageModel.for_inference(model)
rows = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type") == "story"]
pref = []
with model.disable_adapter():           # tắt LoRA -> hành vi base
    for r in rows:
        msgs = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":r["instruction"]}]
        try: p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
        except TypeError: p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(p, return_tensors="pt").to("cuda")
        out = model.generate(**inp, max_new_tokens=400, do_sample=True, temperature=0.8,
                             top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3)
        rejected = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pref.append({"prompt": p, "chosen": r["output"], "rejected": rejected})
with open("preference.jsonl","w",encoding="utf-8") as f:
    for x in pref: f.write(json.dumps(x, ensure_ascii=False)+"\n")
print("preference pairs:", len(pref))
print("VD rejected[:200]:", pref[0]["rejected"][:200])
```

**Cell 7 (ORPO train):**
```python
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig
from unsloth import FastLanguageModel
FastLanguageModel.for_training(model)     # bật lại chế độ train
pref_ds = load_dataset("json", data_files={"train": "preference.jsonl"})["train"]
orpo = ORPOTrainer(
    model=model,
    args=ORPOConfig(
        beta=ORPO_BETA, learning_rate=ORPO_LR, num_train_epochs=ORPO_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        max_length=ORPO_MAX_LEN, max_prompt_length=ORPO_PROMPT_LEN,
        logging_steps=5, output_dir="orpo_out", seed=SEED, warmup_steps=WARMUP_STEPS),
    train_dataset=pref_ds, processing_class=tokenizer,
)
orpo.train()
```

**Cell 8 (sinh thử cuối):**
```python
FastLanguageModel.for_inference(model)
msgs = [{"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":"Viết một truyện ngụ ngôn cho trẻ em về chủ đề: lòng kiên nhẫn. Bài học đạo đức: kiên nhẫn sẽ thành công. Độ tuổi phù hợp: 6-8 tuổi."}]
try: p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
except TypeError: p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tokenizer(p, return_tensors="pt").to("cuda")
out = model.generate(**inp, max_new_tokens=400, do_sample=True, temperature=0.8, top_p=0.9,
                     repetition_penalty=1.3, no_repeat_ngram_size=3)
print(tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Cell 9 (export GGUF q8 + tải về):**
```python
model.save_pretrained_gguf(MERGED_DIR, tokenizer, quantization_method="q8_0")
import glob, os
g = sorted(glob.glob("**/*.gguf", recursive=True), key=lambda p: os.path.getsize(p), reverse=True)
print("GGUF:", [(c, round(os.path.getsize(c)/1e6,1),"MB") for c in g])
from google.colab import files; files.download(g[0])
```

- [ ] **Step 2: Chạy generator + validate nbformat**

Run:
```bash
python3 <scratchpad>/gen_nb_4b.py
python3 -c "import json; nb=json.load(open('notebooks/finetune_qwen3_4b_orpo.ipynb')); print('cells', len(nb['cells']), 'nbformat', nb['nbformat'])"
```
Expected: 10 cells, nbformat 4.

- [ ] **Step 3: Commit**

```bash
git add notebooks/finetune_qwen3_4b_orpo.ipynb
git commit -m "feat: Qwen3-4B multi-stage fine-tune notebook (SFT DoRA+NEFTune -> ORPO)"
```

---

## Task 3: Ollama Modelfile cho gguf 4B

**Files:**
- Modify: `ollama/Modelfile`

- [ ] **Step 1: Cập nhật `ollama/Modelfile`** — đổi FROM sang gguf 4B (giữ system prompt + repeat_penalty):

```dockerfile
FROM ../models/qwen3-4b-fable.Q8_0.gguf

SYSTEM """Bạn là người kể truyện ngụ ngôn cho trẻ em. Bạn chỉ viết truyện ngụ ngôn hư cấu, trong sáng, phù hợp lứa tuổi, luôn có một bài học đạo đức rõ ràng ở cuối."""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
PARAMETER num_ctx 2048
```
(Tên file gguf chính xác xác định ở Task 4 sau khi tải về; đổi cho khớp.)

- [ ] **Step 2: Commit**

```bash
git add ollama/Modelfile
git commit -m "feat: Ollama Modelfile for Qwen3-4B fable-tuned gguf"
```

---

## Task 4: Runbook — chạy notebook trên Colab T4 (tương tác qua colab-mcp)

> Không phải task code. Đây là quy trình chạy + kiểm chứng. Thực hiện khi user sẵn sàng đăng nhập Google.

- [ ] **Step 1: Reconnect Colab** — gọi `open_colab_browser_connection`; user đăng nhập Google + đặt runtime **T4 GPU**. Xác nhận `torch.cuda.is_available()` = True.

- [ ] **Step 2: Đưa dữ liệu vào Colab** — tạo secret gist tạm chứa `data/processed/train.jsonl` + `val.jsonl` (đã lọc), clone trong Colab về `train.jsonl`/`val.jsonl`. (Xóa gist sau khi xong.)

- [ ] **Step 3: Nạp các cell từ notebook** vào phiên Colab và chạy tuần tự: install → hyperparams → model → SFT → preference → ORPO → sinh thử → export.

- [ ] **Step 4: Kiểm chứng**
  - SFT: train/val loss giảm dần.
  - Preference: `preference pairs` > 0, `rejected` có nội dung.
  - ORPO: chạy xong không OOM (nếu OOM → hạ `ORPO_MAX_LEN=1024`, chạy lại cell ORPO).
  - Sinh thử cuối: truyện **mạch lạc, đúng thể loại ngụ ngôn, có bài học, an toàn**. Nếu lan man → hạ `SFT_LR`/`SFT_EPOCHS` hoặc `ORPO_EPOCHS`, train lại.
  Expected: truyện rõ ràng tốt hơn bản 1.7B.

- [ ] **Step 5: Tải GGUF về** (`qwen3-4b-fable*.Q8_0.gguf`, ~4GB) vào `~/Downloads`.

- [ ] **Step 6: Tạo model Ollama + kích hoạt app 2×4B**

Run (local):
```bash
cd "/Users/trieulh/Documents/Master/20252B_IT5410/Final"
mv ~/Downloads/qwen3-4b-fable*.Q8_0.gguf models/    # khớp tên với Modelfile (Task 3)
ollama rm fable-tuned 2>/dev/null; ollama create fable-tuned -f ollama/Modelfile
uvicorn app.main:app --port 8000     # default đã là base=qwen3:4b, tuned=fable-tuned
```
Expected: `/models` trả base=qwen3:4b, tuned=fable-tuned; dropdown app có 2 model 4B; sinh truyện bằng tuned mạch lạc, không `<think>`.

- [ ] **Step 7: Dọn dẹp** — xóa gist tạm; xóa model `fable-tuned` cũ (1.7B) đã bị ghi đè.

---

## Self-Review

**Spec coverage:**
- §2 pipeline 2 tầng (SFT→preference→ORPO→export) → Task 2 (cell 5,6,7,9). ✅
- §3 data (SFT + preference chosen/rejected) → Task 2 cell 4,6. ✅
- §4 tham số (DoRA, NEFTune, responses-only, ORPO LR/beta, để mở) → Task 2 cell 2,3,5,7. ✅
- §5 files (notebook, Modelfile, config base 4B, MODEL_INFO) → Task 1,2,3. ✅
- §1 app 2×4B bỏ 1.7B → Task 1 (config) + Task 4 step 6. ✅
- §6 rủi ro (OOM→seq 1024; lan man→hạ LR/epoch) → Task 4 step 4. ✅
- torchao fix + repetition_penalty (bài học lần trước) → Task 2 cell 1, cell 6/8. ✅

**Placeholder scan:** tên file gguf để mở (khớp ở Task 4) — có chủ đích, đã ghi rõ. Không có TODO/TBD khác. ✅

**Type consistency:** `BASE_MODEL="qwen3:4b"`, `TUNED_MODEL="fable-tuned"` nhất quán Task 1 ↔ Task 4. Biến hyperparams (SFT_LR, USE_DORA, ORPO_*) khai báo cell 2, dùng cell 3/5/7 nhất quán. `preference.jsonl` schema {prompt,chosen,rejected} khớp ORPOTrainer. ✅

**Lưu ý phụ thuộc:** Task 1 (config) + Task 2/3 (notebook/Modelfile) làm & commit được ngay không cần GPU. Task 4 cần Colab + user auth + tải gguf 4B (~4GB).
