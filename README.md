# English Fable Generator

Trình tạo **truyện ngụ ngôn tiếng Anh** cho trẻ em, chạy trên mô hình ngôn ngữ cục bộ (qua Ollama), có **streaming thời gian thực**, **guardrail nhiều lớp**, **observability** đầy đủ và **đánh giá chất lượng theo phương pháp khoa học** (LLM-as-judge có dẫn chứng + metric khách quan). Đồ án môn IT5410.

> **Mô hình chỉ là tham số cấu hình.** Ứng dụng không phụ thuộc một model cụ thể. Ví dụ mặc định dùng `Qwen3-4B-Instruct-2507`, nhưng bạn có thể trỏ sang **bất kỳ model chat nào Ollama phục vụ được** (Llama, Gemma, Mistral, model bạn tự fine-tune...) chỉ bằng cách sửa `config/models.json`. Xem [§7](#7-cấu-hình).

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Kiến trúc](#2-kiến-trúc)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt](#4-cài-đặt)
5. [Chạy hệ thống](#5-chạy-hệ-thống)
6. [Hướng dẫn sử dụng](#6-hướng-dẫn-sử-dụng)
7. [Cấu hình](#7-cấu-hình)
8. [Phương pháp khoa học sử dụng](#8-phương-pháp-khoa-học-sử-dụng)
9. [Cách thức đánh giá](#9-cách-thức-đánh-giá)
10. [Giải thích các thông số trong app](#10-giải-thích-các-thông-số-trong-app)
11. [Kiểm thử](#11-kiểm-thử)
12. [Cấu trúc thư mục](#12-cấu-trúc-thư-mục)
13. [Phase C — Train 200M Transformer (from scratch)](#13-phase-c--train-a-200m-fable-transformer-from-scratch-no-fine-tune)
14. [Xử lý sự cố](#14-xử-lý-sự-cố)
15. [Tài liệu tham khảo & trích dẫn](#15-tài-liệu-tham-khảo--trích-dẫn)

---

## 1. Giới thiệu

Mục tiêu: sinh **truyện ngụ ngôn (fable)** tiếng Anh cho trẻ 4–7 tuổi, có nhân vật (thường là con vật), tình huống, và **bài học đạo đức rõ ràng** ở cuối. Nguồn dữ liệu tham chiếu/huấn luyện đích: [`klusai/ds-tf1-en-3m`](https://huggingface.co/datasets/klusai/ds-tf1-en-3m) (TF1-EN-3M).

Điểm nhấn của đồ án là **tính khoa học của việc huấn luyện và đánh giá**; ứng dụng được thiết kế để **phản ánh trực quan** kết quả đó:

- Sinh truyện **streaming** từng token, quan sát được toàn bộ quá trình.
- **Guardrail 4 lớp** đảm bảo chỉ tạo nội dung trẻ em an toàn.
- **Đánh giá tự động** mỗi truyện theo 4 trục có **dẫn chứng trích từ truyện**.
- **Chế độ so sánh** (Compare) 2 model đặt cạnh nhau (vd base vs fine-tuned).
- **Tab Results** trình bày đánh giá khoa học theo lô (batch).

---

## 2. Kiến trúc

```
  Trình duyệt (React + TypeScript + Astryx + recharts)
        │  GET /models, POST /generate/stream (SSE), POST /evaluate, GET /results
        ▼
  FastAPI backend (app/)
        │  HTTP /api/chat (stream)
        ▼
  Ollama (localhost:11434)  ──►  <model cấu hình được>  (vd Qwen3-4B-Instruct-2507)
```

| Thành phần | Vai trò |
|---|---|
| `app/main.py` | Các endpoint + phục vụ `web/dist` nếu đã build |
| `app/ollama_client.py` | Gọi Ollama (stream + non-stream), forward tham số sinh |
| `app/guardrail/` | Bộ lọc đầu vào/đầu ra (xem [§8](#8-phương-pháp-khoa-học-sử-dụng)) |
| `app/judge.py` | LLM-as-judge 4 trục, trả điểm + lý do dẫn chứng |
| `app/prompt_en.py` | System prompt + dựng prompt tiếng Anh từ 5 yếu tố tường thuật |
| `app/models_registry.py` | Đọc `config/models.json` (model-agnostic) |
| `web/` | Giao diện React (Playground + Results) |

---

## 3. Yêu cầu hệ thống

| Thành phần | Phiên bản khuyến nghị | Ghi chú |
|---|---|---|
| Python | ≥ 3.11 | Backend + test |
| Node.js | ≥ 20 | Build/chạy frontend |
| Ollama | ≥ 0.30 | Phục vụ model cục bộ |
| RAM | ≥ 8 GB | Đủ chạy model ~4B Q8 |
| Đĩa | ~5 GB/model | Cho model GGUF Q8 |

Cài Ollama: https://ollama.com/download (macOS: `brew install ollama`).

---

## 4. Cài đặt

### 4.1. Lấy source

```bash
git clone https://github.com/tungd/tinystory-vn.git
cd tinystory-vn
```

### 4.2. Backend (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev]"             # fastapi, uvicorn, httpx, pydantic, pytest
# Apple Silicon, để phục vụ model 64M qua MLX:
pip install -e ".[dev,inference]"
```

### 4.3. Chuẩn bị một model trong Ollama

> Ứng dụng chạy với **bất kỳ model chat nào của Ollama**. Dưới đây `Qwen3-4B-Instruct-2507` chỉ là **ví dụ**. Lưu ý quan trọng: **nên dùng model dạng *Instruct* (không phải *thinking*)** — model "thinking" (vd `qwen3:4b`) sẽ chèn phần suy luận vào truyện.

**Cách A — kéo model có sẵn từ Ollama registry:**

```bash
ollama pull llama3.2:3b-instruct-fp16     # ví dụ 1
# hoặc
ollama pull gemma2:2b                       # ví dụ 2
```

**Cách B — tạo model từ file GGUF cục bộ** (ví dụ Qwen3-4B-Instruct-2507):

```bash
cat > /tmp/Modelfile.instruct <<EOF
FROM $(pwd)/models/qwen3-4b-instruct-2507.Q8_0.gguf
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
PARAMETER num_ctx 4096
EOF
ollama create qwen3-4b-instruct -f /tmp/Modelfile.instruct
```

Sau đó **khai báo model đó trong `config/models.json`** (xem [§7.1](#71-model-registry--configmodelsjson)) — trường `ollama` phải khớp tên trong `ollama list`.

### 4.4. Frontend (Node)

```bash
cd web
npm install
npm run build        # tạo web/dist để backend phục vụ
cd ..
```

---

## 5. Chạy hệ thống

Đảm bảo **Ollama đang chạy** (`ollama serve` hoặc app Ollama đang mở).

### Cách 1 — Một cổng duy nhất (khuyên dùng cho demo)

Backend tự phục vụ `web/dist`. Sau khi đã `npm run build`:

```bash
source .venv/bin/activate
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt: **http://127.0.0.1:8000**

> Backend chỉ mount `web/dist` lúc khởi động. Build lại frontend thì **khởi động lại** uvicorn. Sửa code backend cần khởi động lại (hoặc chạy với `--reload`).

### Cách 2 — Chế độ phát triển (hot reload)

```bash
# Terminal 1 — backend
source .venv/bin/activate
python3 -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (Vite dev, proxy /api → :8000)
cd web && npm run dev
```

---

## 6. Hướng dẫn sử dụng

### Tab Playground

**Single mode** — nhập 5 yếu tố tường thuật (đều **tùy chọn**; để trống thì model tự quyết), các ô hỗ trợ **nhiều dòng**:

| Ô nhập | Ý nghĩa | Ví dụ |
|---|---|---|
| Main character | Nhân vật chính | *a clever fox* |
| Setting | Bối cảnh | *a foggy marsh* |
| Challenge | Vấn đề/xung đột | *a hungry heron guards the only fish* |
| Outcome | Cách giải quyết/kết cục | *he tricks the heron and escapes* |
| Teaching | Bài học/moral | *cleverness beats brute force* |

Chọn **độ dài** (Short/Medium/Long), **model**, bật/tắt **guardrail**, bấm **Generate fable**. Có sẵn **presets** (fable kinh điển) và nút **Surprise me**.

- **Cột giữa**: truyện hiện dần (guardrail OFF) hoặc "Processing…" rồi ra truyện (guardrail ON), render markdown an toàn.
- **Cột phải**: **Activity Log** (các bước xử lý theo lớp) + **Observability** (xem [§10](#10-giải-thích-các-thông-số-trong-app)).
- **Quick Evaluation**: tự động chấm 4 trục sau khi truyện hiện ra (không chặn UI), kèm **radar + bảng điểm + lý do dẫn chứng** cho từng trục.

**Compare mode** — chọn 2 model đặt cạnh nhau, mỗi bên stream riêng, kèm khung **Verdict** (radar overlay + delta + xếp hạng). Chỉ bật khi registry có ≥ 2 model.

### Tab Results

Trình bày **đánh giá khoa học theo lô** (batch) đọc từ `results/eval_summary.json`. Khi chưa có file, hiện placeholder — bình thường. File này được sinh bởi **Notebook B** (`notebooks/eval_gen_fable200m_colab.ipynb`) chạy trên Colab qua `google-colab-cli` (xem [§13](#13-phase-c--train-a-200m-fable-transformer-from-scratch-no-fine-tune) và `docs/runbooks/colab-train.md`). Xem [§9](#9-cách-thức-đánh-giá).

---

## 7. Cấu hình

### 7.1. Model registry — `config/models.json`

Danh sách model app hiển thị/dùng được. Đây là nơi **duy nhất** cần sửa để đổi/thêm model:

```json
[
  {
    "id": "base-qwen3-4b",
    "name": "Qwen3-4B-Instruct-2507",
    "ollama": "qwen3-4b-instruct",
    "kind": "base",
    "desc": "Base model (Instruct-2507, non-thinking), chưa fine-tune"
  }
]
```

| Trường | Ý nghĩa |
|---|---|
| `id` | Định danh dùng trong API (`model_id`) |
| `name` | Tên hiển thị trên UI |
| `ollama` | Tên model trong Ollama (khớp `ollama list`) |
| `kind` | `base` hoặc `finetuned` |
| `desc` | Mô tả ngắn (tooltip) |

Thêm model fine-tune: thêm một phần tử `kind: "finetuned"` → tự xuất hiện trong dropdown và **bật được Compare mode**.

### 7.2. Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Địa chỉ Ollama |
| `FABLE_MODELS_PATH` | `config/models.json` | Đường dẫn registry |
| `FABLE_JUDGE_MODEL_ID` | `base-qwen3-4b` | `id` model dùng để chấm điểm |
| `FABLE_RESULTS_PATH` | `results/eval_summary.json` | File kết quả batch eval |
| `FABLE_THINK` | `false` | Bật/tắt thinking (giữ `false`) |
| `FABLE_TIMEOUT` | `120` | Timeout gọi Ollama (giây) |
| `GEN_TEMPERATURE` | `0.8` | Nhiệt độ sinh |
| `GEN_TOP_P` | `0.9` | top-p |
| `GEN_REPEAT_PENALTY` | `1.3` | Phạt lặp (tránh vòng lặp) |

---

## 8. Phương pháp khoa học sử dụng

### 8.1. Cấu trúc tường thuật (Narrative Structure — TF1)

Đầu vào 5 ô bám theo cấu trúc kể chuyện của dataset TF1-EN-3M: **Main Character, Setting, Challenge, Outcome, Teaching**. Đây là các biến điều khiển để prompt có cấu trúc, đồng thời là căn cứ chấm trục **Prompt Adherence**.

### 8.2. Guardrail 4 lớp

Cơ chế bảo vệ đảm bảo app chỉ tạo truyện ngụ ngôn trẻ em an toàn:

| Lớp | Vị trí | Kiểm tra |
|---|---|---|
| **Layer 1** | Đầu vào (`check_input_en`) | Lọc từ cấm (profanity), prompt-injection, yêu cầu viết mã, và **nội dung không dành cho trẻ em**: người lớn/giới hạn tuổi (18+, nsfw, x/r-rated, "adult/mature content"), tình dục (sexual/erotic/nude), bạo lực đồ họa (gore). Từ chối sớm trước khi gọi model. |
| **Layer 2–3** | Sinh có kiểm soát | System prompt ràng buộc "chỉ viết fable trẻ em", tham số sinh (temperature/top_p/repeat_penalty), thử lại nếu cần |
| **Layer 4** | Đầu ra (`check_output_en`) | Quét từ cấm + nội dung rỗng trong truyện đã sinh trước khi trả về |

> Từ đơn như "adult"/"grown-up" **không** bị chặn (fable có thể có con vật trưởng thành, vd "an adult lion"); chỉ chặn khi có dấu hiệu giới hạn tuổi (18+) hoặc cụm "adult/mature + content/theme/lesson…".

**Bất biến quan trọng**: khi guardrail **ON**, app **không** stream token thô — chỉ phát log các bước và trả **truyện cuối đã lọc** (đảm bảo không lộ nội dung chưa kiểm duyệt). Mọi lần chặn đều được ghi vào Activity Log kèm **lớp + loại vi phạm** (vd `Layer 1 BLOCKED [out_of_scope]`, `Layer 4 BLOCKED`).

### 8.3. Phương pháp đánh giá (bám ADR-0002)

Đánh giá **không tự chế tiêu chí**, dùng đúng bộ phương pháp của paper TF1-EN-3M ([Nadas et al., 2025](#15-tài-liệu-tham-khảo--trích-dẫn)) + metric kinh điển:

1. **Metric khách quan (reference-free):** Perplexity (trên test held-out), Distinct-1/2, Self-BLEU, Flesch Reading Ease.
2. **LLM-as-judge panel:** ≥ 2–3 model **khác họ** chấm **4 trục** (thang 1–10), không thêm/bớt trục — đúng bộ trục *grammar, creativity, moral clarity, template/prompt adherence* của paper.
3. **Độ tin cậy judge:** weighted **Cohen's κ** + **Kendall's τ**; kết luận before/after **theo THỨ HẠNG** (đa số judge xếp model nào cao hơn), không dựa điểm tuyệt đối của 1 judge.

> Bộ 4 trục LLM-as-judge, các metric đa dạng/độ đọc dễ (Distinct/Self-BLEU/Flesch) và cách dùng panel judge khác họ trong đồ án này **kế thừa trực tiếp phương pháp đánh giá của paper TF1-EN-3M** ([Nadas et al., 2025](#15-tài-liệu-tham-khảo--trích-dẫn)). Chi tiết quyết định: `docs/adr/0002-evaluation-methodology.md`.

---

## 9. Cách thức đánh giá

Có **hai tầng** đánh giá, phục vụ mục đích khác nhau:

| | **Quick Evaluation** (Playground) | **Batch Evaluation** (tab Results) |
|---|---|---|
| Phạm vi | 1 truyện vừa sinh | Cả tập test held-out |
| Judge | 1 judge (chỉ báo nhanh) | Panel ≥ 2–3 judge khác họ |
| Nội dung | 4 trục + **lý do dẫn chứng** + radar | Metric khách quan + judge panel + κ/τ + loss + kết luận theo rank |
| Nguồn | gọi `POST /evaluate` tức thời | Notebook B trên Colab (`notebooks/eval_gen_fable200m_colab.ipynb`) → `results/eval_summary.json` |
| Vai trò | Xem nhanh trên UI | **Số liệu chuẩn cho báo cáo** |

**4 trục chấm** (thang 0–10 trên UI; 1–10 theo paper):

| Trục | Ý nghĩa |
|---|---|
| **Grammar** | Độ đúng ngữ pháp/mạch lạc câu chữ |
| **Creativity** | Độ sáng tạo/hấp dẫn của truyện |
| **Moral Clarity** | Bài học có rõ ràng, truyền tải tốt không |
| **Prompt Adherence** | Bám sát các yếu tố tường thuật người dùng nhập |
| *Overall* | Trung bình 4 trục |

Judge trả kèm **lý do ngắn có trích dẫn cụ thể từ truyện** cho từng trục (giúp con số có căn cứ, không "trần trụi"). Để giảm self-bias, nên chọn judge **khác** model sinh.

---

## 10. Giải thích các thông số trong app

### 10.1. Observability (panel bên phải, mỗi lần sinh)

| Thông số | Ý nghĩa |
|---|---|
| **Model** | Model + loại (base/finetuned) đã dùng |
| **Temperature** | Độ ngẫu nhiên khi lấy mẫu (cao = đa dạng/rủi ro lạc đề hơn) |
| **Top P** | Nucleus sampling: chỉ lấy trong khối xác suất tích lũy p |
| **Repetition penalty** | Phạt token lặp (tránh truyện bị lặp vòng) |
| **Max tokens** (`num_predict`) | Giới hạn token sinh (theo độ dài Short/Medium/Long) |
| **Seed** | Hạt ngẫu nhiên cố định → **tái lập** kết quả |
| **Input tokens** | Số token prompt đưa vào model |
| **Output tokens** | Số token model sinh ra |
| **Latency** | Thời gian sinh (ms) |
| **Tokens / sec** | Tốc độ sinh (output tokens / thời gian) |
| **Prompt sent** | Prompt **thực tế** đã gửi cho model (bung ra xem được) |

### 10.2. Activity Log

Dòng thời gian chi tiết các bước xử lý, mỗi bước có trạng thái (running/ok/blocked) + mô tả + timestamp:

| Bước | Nội dung log |
|---|---|
| **Prepare request** | Độ dài fable, số ký tự prompt, giới hạn token |
| **Model config** | Tên model + loại (base/finetuned) + tag Ollama + tham số (temperature/top_p/repeat_penalty/seed) |
| **Input check (Layer 1)** | Kết quả quét đầu vào; nếu chặn: `Layer 1 BLOCKED [<loại>]: <lý do>` |
| **Generating (Layer 2-3)** | Khi xong: số token sinh, thời gian, tokens/giây, số prompt token |
| **Output check (Layer 4)** | Kết quả quét đầu ra; nếu chặn: `Layer 4 BLOCKED: <lý do>` (+ số lần thử lại còn lại) |

Nhờ vậy có thể quan sát đầy đủ **model đã dùng gì, sinh nhanh ra sao, và guardrail chặn ở đâu**.

> **Khóa nhập liệu khi đang sinh**: trong lúc generate, toàn bộ Presets/Surprise me, 5 ô nhập, độ dài, model, guardrail bị **disable** và nút chuyển thành **"Generating…"** để tránh sửa giữa chừng.

### 10.3. Độ dài truyện

| Mức | Số từ mục tiêu | `num_predict` |
|---|---|---|
| Short | ~120–180 từ | 300 |
| Medium | ~250–350 từ | 600 |
| Long | ~450–600 từ | 1000 |

---

## 11. Kiểm thử

```bash
# Backend (pytest)
source .venv/bin/activate
python3 -m pytest -q

# Frontend (type-check nghiêm ngặt + build)
cd web && npm run build
```

---

## 12. Cấu trúc thư mục

```
.
├── app/                    # Backend FastAPI
│   ├── main.py             # Endpoints + serve web/dist
│   ├── ollama_client.py    # Gọi Ollama (stream/non-stream)
│   ├── guardrail/          # Bộ lọc 4 lớp (tiếng Anh): input_filter, output_filter, wordlist
│   ├── judge.py            # LLM-as-judge 4 trục + rationale
│   ├── prompt_en.py        # System prompt + build prompt + build_seed_prompt (EN)
│   ├── models_registry.py  # Đọc config/models.json (model-agnostic)
│   └── config.py           # Biến môi trường + tham số sinh
├── config/models.json      # Model registry (base-qwen3-4b, fable-200m)
├── runs/                   # Canonical v1/v2/v3 data, artifacts, logs, results
│   ├── v1/                 # 29M baseline + full local resume state
│   ├── v2/                 # 64M Metaspace model + recovered logs/results
│   └── v3/                 # Isolated conditioning-focused run (planned)
├── web/                    # Frontend React + Astryx + recharts
│   └── src/                # App.tsx, components/, api.ts
├── notebooks/              # Colab training/eval (chạy qua google-colab-cli)
│   ├── train_fable200m_colab.ipynb     # Notebook A: train → checkpoint (Drive)
│   └── eval_gen_fable200m_colab.ipynb  # Notebook B: eval + gen → eval_summary.json
├── scripts/               # Tiện ích dữ liệu + train (chạy trên Colab)
│   ├── prepare_tf1.py      # stream TF1 → BPE fables.jsonl + tokenizer.json (hoặc char-mode)
│   ├── fable_tokenizer.py  # char-level tokenizer fallback
│   ├── metrics.py          # Distinct-1/2, Self-BLEU, Flesch
│   ├── train_local.py      # single-script train→gen→eval_summary.json (nháp pipeline)
│   └── smoke_train.py      # tiny local smoke test
├── tests/                  # pytest (backend)
├── docs/
│   ├── adr/                # 0002-evaluation-methodology, 0003-from-scratch-200m
│   ├── runbooks/colab-train.md   # google-colab-cli runbook (canonical train path)
│   └── superpowers/plans/2026-07-08-keyword-guided-fable-generation.md
└── models/                 # Compatibility links into runs/*/artifacts
```

> Lịch sử: đồ án từng fine-tune Qwen3-4B (notebook `finetune_qwen3_*`, ADR-0001) và
> sinh dữ liệu tiếng Việt (`scripts/extract_new_fables.py`, `prepare_data.py`,
> `app/prompt.py`). Các file này đã bị xoá — quyết định train **từ đầu** 200M trên
> TF1-EN-3M (ADR-0003) và output tiếng Anh.

---

## 13. Phase C — Train a 200M Fable Transformer (from scratch, no fine-tune)

Đồ án **không được phép fine-tune** base model. Thay vào đó ta **train từ đầu** một
transformer ~200M tham số trên TF1-EN-3M, sinh truyện theo **keyword guidance** =
hai seed: **nhân vật chính** + **bài học đạo đức** (không RAG, không base model).

- **Keyword guidance** = prefix điều khiển: `<char> {character} </char>\n<moral> {moral} </moral>\n<story>\n`. Model học tiếp nối từ prefix để sinh thân truyện. Format này được dùng nhất quán ở `scripts/prepare_tf1.format_sample`, `app/prompt_en.build_seed_prompt` và notebook Colab.
- **Notebook A** `notebooks/train_fable200m_colab.ipynb`: chuẩn bị data (stream TF1 → BPE tokenizer → `fables.jsonl` + `tokenizer.json`) → train GPT2LMHeadModel (~200M, `transformers.Trainer`) → lưu checkpoint lên Drive.
- **Notebook B** `notebooks/eval_gen_fable200m_colab.ipynb`: load checkpoint → sinh truyện từ seed → metric khách quan (Distinct-1/2, Self-BLEU, Flesch) + LLM-as-judge 4 trục → `eval_summary.json`.
- **Chạy trên Colab qua `google-colab-cli`** (xem `docs/runbooks/colab-train.md`): `uv tool install google-colab-cli` → `colab new -s trainer --gpu T4` → `colab upload` scripts → `colab exec -f notebooks/...ipynb` → `colab download` checkpoint → `colab stop`. Đây là đường đi **chuẩn** (canonical) của đồ án; mọi bước train/eval đều chạy trên Colab, không train trên máy local.
- **Dùng trong app**: checkpoint 63M đã được chuyển sang `models/fable-64m-mlx`; chạy `mlx_lm.server` và đặt `FABLE_BACKEND=openai`, `OLLAMA_BASE_URL=http://127.0.0.1:8080`. Entry `fable-200m` trong `config/models.json` trỏ tới model này; model ID thực tế được tự phát hiện qua `/v1/models`.
- **Script hỗ trợ (cùng chạy trên Colab, không local)**: `scripts/prepare_tf1.py` (stream TF1 → BPE `fables.jsonl` + `tokenizer.json`, hoặc char-mode fallback), `scripts/fable_tokenizer.py`, `scripts/metrics.py` (Distinct/Self-BLEU/Flesch), `scripts/train_local.py` (single-script train→gen→`eval_summary.json` dùng làm nháp kiểm thử pipeline), `scripts/smoke_train.py` (tiny local smoke test). Notebook Colab import trực tiếp các script này.
- **Tiếp tục một lượt train** (resume): xem checklist ở `docs/runbooks/colab-train.md`. Tóm tắt: đảm bảo `google-colab-cli` đã auth (`colab sessions` trả danh sách), `colab new -s trainer --gpu T4` (hoặc `--keep` để giữ VM giữa các bước), `colab upload` 3 scripts (`prepare_tf1.py`, `fable_tokenizer.py`, `metrics.py`), `colab exec -f notebooks/train_fable200m_colab.ipynb`, `colab download` checkpoint về `models/`, `colab stop`. Nếu cần scale lên 200k fables / nhiều epoch, đổi `--gpu L4|A100` hoặc tăng `N_FABLES`/`max_steps` trong notebook.
- Kế hoạch chi tiết: `docs/superpowers/plans/2026-07-08-keyword-guided-fable-generation.md`; quyết định: `docs/adr/0003-from-scratch-200m.md`.

---

## 14. Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách khắc phục |
|---|---|
| Truyện ra toàn văn "Okay, the user wants…" | Đang dùng model *thinking*. Chọn model dạng **Instruct** (xem [§4.3](#43-chuẩn-bị-một-model-trong-ollama)). |
| `GET /models` rỗng hoặc lỗi | Kiểm tra `config/models.json` hợp lệ + đúng `FABLE_MODELS_PATH`. |
| Sinh treo / timeout | Ollama chưa chạy (`ollama serve`), hoặc `ollama` trong registry chưa tồn tại (`ollama list`). Tăng `FABLE_TIMEOUT`. |
| Frontend không cập nhật sau build | Khởi động lại uvicorn (mount `web/dist` chỉ lúc khởi động). |
| Compare mode bị mờ (disabled) | Registry chỉ có 1 model. Thêm model thứ 2 vào `config/models.json`. |
| Điểm eval về 0 hết | Judge model trả JSON hỏng nặng; thử judge khác qua `FABLE_JUDGE_MODEL_ID`. |
| Port 8000 bận | Chạy uvicorn `--port <khác>` (cập nhật proxy Vite nếu dùng dev mode). |

---

## 15. Tài liệu tham khảo & trích dẫn

Đồ án này sử dụng **dataset** và **phương pháp đánh giá** của paper TF1-EN-3M. Nếu bạn dùng lại, vui lòng trích dẫn:

> Nadas, M., Diosan, L., Piscoran, A., & Tomescu, A. (2025). *TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small, Open Language Models*. arXiv:2504.20605.

```bibtex
@article{nadas2025tf1en3m,
  title   = {TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small, Open Language Models},
  author  = {Nadas, Mihai and Diosan, Laura and Piscoran, Andrei and Tomescu, Andreea},
  journal = {arXiv preprint arXiv:2504.20605},
  year    = {2025}
}
```

- **Dataset:** [`klusai/ds-tf1-en-3m`](https://huggingface.co/datasets/klusai/ds-tf1-en-3m) (CC BY 4.0)
- **Paper:** https://arxiv.org/abs/2504.20605

**Phương pháp kế thừa từ paper** (xem [§8.3](#83-phương-pháp-đánh-giá-bám-adr-0002) và [§9](#9-cách-thức-đánh-giá)): bộ 4 trục LLM-as-judge (grammar, creativity, moral clarity, template/prompt adherence), panel judge từ **các họ model khác nhau**, và các **metric reference-free** về đa dạng/độ đọc dễ (Distinct-1/2, Self-BLEU, Flesch Reading Ease).
