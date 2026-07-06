# English Fable Generator

Trình tạo **truyện ngụ ngôn tiếng Anh** cho trẻ em, chạy trên mô hình ngôn ngữ cục bộ (Ollama), có **streaming**, **guardrail nhiều lớp**, **observability** đầy đủ và **đánh giá chất lượng khoa học** (LLM-as-judge + metric khách quan). Đồ án IT5410.

- **Backend**: FastAPI (SSE streaming) + Ollama.
- **Frontend**: React + TypeScript + Vite + Astryx design system + recharts.
- **Mô hình nền (base)**: `Qwen3-4B-Instruct-2507` (non-thinking) phục vụ qua Ollama.
- **Dữ liệu huấn luyện đích**: [`klusai/ds-tf1-en-3m`](https://huggingface.co/datasets/klusai/ds-tf1-en-3m) (TF1-EN-3M) — dùng cho Phase C fine-tune (hiện hoãn).

---

## 1. Kiến trúc tổng quan

```
  Trình duyệt (React + Astryx)
        │  fetch /models, POST /generate/stream (SSE), POST /evaluate, GET /results
        ▼
  FastAPI backend (app/)
        │  HTTP /api/chat (stream)
        ▼
  Ollama (localhost:11434)  ──►  qwen3-4b-instruct (base, non-thinking)
```

- `app/main.py` — các endpoint: `GET /models`, `POST /generate/stream`, `POST /evaluate`, `GET /results`; tự phục vụ `web/dist` nếu đã build.
- `app/guardrail/` — bộ lọc đầu vào/đầu ra 4 lớp (tiếng Anh).
- `app/judge.py` — LLM-as-judge 4 trục (Grammar, Creativity, Moral Clarity, Prompt Adherence, 0-10).
- `app/ollama_client.py` — gọi Ollama (stream + non-stream), forward các tham số sinh.
- `config/models.json` — **model registry** (thêm model mới ở đây).
- `web/` — ứng dụng React (Playground + Results).

---

## 2. Yêu cầu hệ thống (Prerequisites)

| Thành phần | Phiên bản khuyến nghị | Ghi chú |
|---|---|---|
| Python | ≥ 3.11 | Chạy backend + test |
| Node.js | ≥ 20 (đã test v25) | Build/chạy frontend |
| Ollama | ≥ 0.30 | Phục vụ mô hình cục bộ |
| Dung lượng đĩa | ~5 GB | Cho model base (GGUF Q8 ~4 GB) |
| RAM | ≥ 8 GB | Đủ để chạy Qwen3-4B Q8 |

Cài Ollama: xem https://ollama.com/download (macOS: `brew install ollama`).

---

## 3. Cài đặt (Installation)

### 3.1. Lấy source

```bash
git clone <repo-url>
cd Final
```

### 3.2. Backend (Python)

Tạo virtualenv rồi cài package ở chế độ editable (deps khai báo trong `pyproject.toml`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # fastapi, uvicorn, httpx, pydantic, pytest
```

> Nếu không muốn virtualenv, có thể cài trực tiếp: `pip install "fastapi>=0.111" "uvicorn[standard]>=0.30" "httpx>=0.27" "pydantic>=2.7"`.

### 3.3. Mô hình nền trong Ollama (QUAN TRỌNG)

Ứng dụng cần một model Ollama tên `qwen3-4b-instruct`. **Phải dùng bản Instruct-2507 (non-thinking)** — KHÔNG dùng `qwen3:4b` mặc định, vì đó là model *thinking* và sẽ trả về phần suy luận (reasoning) thay vì truyện.

**Cách A — tạo từ file GGUF cục bộ** (nếu bạn đã có `models/qwen3-4b-instruct-2507.Q8_0.gguf`):

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

**Cách B — kéo từ Ollama registry** (cần mạng):

```bash
ollama pull qwen3:4b-instruct-2507   # hoặc tag instruct tương đương
# rồi trỏ registry sang tag đó (sửa "ollama" trong config/models.json)
```

Kiểm tra:

```bash
ollama list | grep qwen3-4b-instruct
ollama serve        # nếu Ollama chưa chạy nền (macOS app tự chạy)
```

### 3.4. Frontend (Node)

```bash
cd web
npm install
npm run build       # tạo web/dist để backend phục vụ
cd ..
```

---

## 4. Chạy hệ thống (Running)

Đảm bảo **Ollama đang chạy** (`ollama serve` hoặc app Ollama đang mở).

### Cách 1 — Một cổng duy nhất (khuyên dùng cho demo)

Backend tự phục vụ `web/dist` tại `/`. Sau khi đã `npm run build`:

```bash
source .venv/bin/activate
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt: **http://127.0.0.1:8000**

> Lưu ý: backend chỉ mount `web/dist` lúc khởi động. Nếu bạn build lại frontend, hãy **khởi động lại** uvicorn.

### Cách 2 — Chế độ phát triển (hot reload)

Hai terminal:

```bash
# Terminal 1 — backend
source .venv/bin/activate
python3 -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (Vite dev, proxy /api → :8000)
cd web && npm run dev
```

Mở trình duyệt tại URL Vite in ra (mặc định http://localhost:5173).

---

## 5. Sử dụng (Usage)

### Tab Playground

- **Single mode**: nhập 5 yếu tố tường thuật (Main character, Setting, Challenge, Outcome, Teaching — đều tùy chọn, để trống thì model tự quyết), chọn độ dài, chọn model, bật/tắt guardrail, bấm **Generate fable**.
  - Cột giữa: truyện stream dần (guardrail OFF) hoặc hiện "Processing…" rồi ra truyện (guardrail ON).
  - Cột phải: **Activity Log** (các bước xử lý theo lớp) + **Observability** (tham số sinh, seed, prompt thực gửi, số token, latency, tokens/sec).
  - **Quick Evaluation**: tự động chấm 4 trục (radar + bảng) sau khi truyện hiện ra (non-blocking).
- **Compare mode**: chọn 2 model đặt cạnh nhau, mỗi bên stream riêng, kèm **Verdict** (radar overlay + delta + xếp hạng). *Chỉ bật khi registry có ≥ 2 model.*
- **Presets** + **Surprise me**: điền nhanh các fable kinh điển.

### Tab Results

Đọc `results/eval_summary.json` (kết quả batch eval khoa học). Khi chưa có file, hiện placeholder — đây là trạng thái bình thường khi mới chạy base model. File này được sinh ở Phase C (`scripts/eval_tf1.py`).

---

## 6. Cấu hình (Configuration)

### 6.1. Model registry — `config/models.json`

Mỗi phần tử:

```json
{
  "id": "base-qwen3-4b",
  "name": "Qwen3-4B-Instruct-2507",
  "ollama": "qwen3-4b-instruct",
  "kind": "base",
  "desc": "Base model (Instruct-2507, non-thinking), chưa fine-tune"
}
```

- `id`: định danh dùng trong API (`model_id`).
- `ollama`: tên model trong Ollama (`ollama list`).
- `kind`: `base` hoặc `finetuned`.
- Thêm model fine-tune sau này: thêm một phần tử mới với `kind: "finetuned"` → tự xuất hiện trong dropdown và bật được Compare mode.

### 6.2. Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Địa chỉ Ollama |
| `FABLE_MODELS_PATH` | `config/models.json` | Đường dẫn registry |
| `FABLE_JUDGE_MODEL_ID` | `base-qwen3-4b` | Model dùng để chấm điểm |
| `FABLE_RESULTS_PATH` | `results/eval_summary.json` | File kết quả batch eval |
| `FABLE_THINK` | `false` | Bật/tắt thinking (giữ `false`) |
| `FABLE_TIMEOUT` | `120` | Timeout gọi Ollama (giây) |
| `GEN_TEMPERATURE` | `0.8` | Nhiệt độ sinh |
| `GEN_TOP_P` | `0.9` | top-p |
| `GEN_REPEAT_PENALTY` | `1.3` | Phạt lặp (tránh vòng lặp) |

---

## 7. Kiểm thử (Testing)

```bash
# Backend (pytest)
source .venv/bin/activate
python3 -m pytest -q               # 49 test

# Frontend (type-check + build)
cd web && npm run build            # tsc strict + vite build
```

---

## 8. Cấu trúc thư mục

```
Final/
├── app/                    # Backend FastAPI
│   ├── main.py             # Endpoints + serve web/dist
│   ├── ollama_client.py    # Gọi Ollama (stream/non-stream)
│   ├── guardrail/          # Bộ lọc 4 lớp (tiếng Anh)
│   ├── judge.py            # LLM-as-judge 4 trục
│   ├── prompt_en.py        # System prompt + build prompt tiếng Anh
│   ├── models_registry.py  # Đọc config/models.json
│   └── config.py           # Biến môi trường + tham số sinh
├── config/models.json      # Model registry
├── web/                    # Frontend React + Astryx
│   └── src/                # App.tsx, components/, api.ts
├── tests/                  # pytest (backend)
├── scripts/                # Tiện ích dữ liệu (Phase C)
├── docs/                   # Spec, plan, ADR
└── models/                 # GGUF cục bộ (gitignored)
```

---

## 9. Phase C — Fine-tune (hiện HOÃN)

Kế hoạch huấn luyện Qwen3-4B trên TF1-EN-3M (SFT → ORPO trên Colab) và batch eval khoa học (perplexity, Distinct-1/2, Self-BLEU, Flesch + LLM-judge panel + Cohen's κ / Kendall's τ) nằm trong `docs/superpowers/plans/` và `docs/adr/0002-evaluation-methodology.md`. Ứng dụng hiện chạy hoàn chỉnh trên **base model**; khi có model fine-tune, chỉ cần thêm vào `config/models.json` để so sánh before/after trong Compare mode + Results tab.

---

## 10. Xử lý sự cố (Troubleshooting)

| Triệu chứng | Nguyên nhân / cách khắc phục |
|---|---|
| Truyện ra toàn văn "Okay, the user wants…" | Đang dùng model *thinking* (`qwen3:4b`). Tạo/đổi sang `qwen3-4b-instruct` (mục 3.3). |
| `GET /models` rỗng hoặc lỗi | Kiểm tra `config/models.json` hợp lệ; đúng đường dẫn `FABLE_MODELS_PATH`. |
| Sinh treo / timeout | Ollama chưa chạy (`ollama serve`), hoặc model chưa tạo (`ollama list`). Tăng `FABLE_TIMEOUT`. |
| Frontend không thấy dữ liệu mới sau build | Khởi động lại uvicorn (mount `web/dist` chỉ lúc khởi động). |
| Compare mode bị mờ (disabled) | Registry chỉ có 1 model. Thêm model thứ 2 vào `config/models.json`. |
| Port 8000 bận | Chạy uvicorn với `--port <khác>` (và cập nhật proxy Vite nếu dùng dev mode). |
