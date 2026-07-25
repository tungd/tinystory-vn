# Báo cáo Kỹ thuật: Tích hợp và Fine-tune Llama 3.2 3B (QLoRA)

**Thư mục làm việc:** `hoangndl/`
**Môi trường:** Kaggle (GPU T4) & Trạm máy cục bộ (Ollama)

---

## 1. Tổng quan

Báo cáo trình bày chi tiết quá trình triển khai fine-tune mô hình **Llama 3.2 3B** bằng phương pháp **QLoRA** (qua thư viện **Unsloth**) trên nền tảng Kaggle. Mục tiêu nhằm ép mô hình học cấu trúc sinh truyện ngụ ngôn từ tập dữ liệu `klusai/ds-tf1-en-3m`, sau đó lượng tử hóa (quantization) về định dạng **GGUF** để tích hợp vào ứng dụng web cục bộ thông qua **Ollama**.

---

## 2. Thông số Kỹ thuật & Cấu hình Huấn luyện

### 2.1. Cấu hình LoRA (PEFT)

| Tham số | Giá trị |
|---|---|
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0.0 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — toàn bộ 7 layer Attention & MLP |

### 2.2. Dữ liệu & Tham số `SFTTrainer`

- **Dataset:** `klusai/ds-tf1-en-3m` — lấy mẫu ngẫu nhiên 1.000 records, chia 90% Train (900 mẫu) / 10% Val (100 mẫu)
- **Batch size hiệu dụng:** 8 (`per_device_train_batch_size = 4` × `gradient_accumulation = 2`)
- **Learning rate:** `2e-4`
- **Tối ưu VRAM:** `use_gradient_checkpointing="unsloth"`, `load_in_4bit=True`, kèm cơ chế **Padding-free**

---

## 3. Nhật ký Thực thi & Kết quả (Logs)

Quá trình huấn luyện được chia làm 2 phiên bản để đối chiếu (**Ablation Study**) về tác động của số Epoch:

| Phiên bản | Số Epoch | Tổng Steps | Thời gian Train | Training Loss (cuối) | Định dạng xuất |
|---|---|---|---|---|---|
| Fable-300 | 1 | ~113 | ~15 phút | ~0.492 | GGUF (Q4_K_M) |
| Fable-1000 | 3 | 339 | ~47 phút | 0.514 | GGUF (Q4_K_M) |

**Chi tiết log — phiên bản 3 Epoch:**
- Tham số khả huấn: **24.313.856** (chiếm **0.75%** tổng model)
- Loss dao động ổn định quanh mức **0.4 – 0.5**
- Validation Loss tại epoch cuối: **0.451**
- Convert sang GGUF Q4_K_M bằng `llama.cpp`: thành công, không gặp lỗi

---

## 4. Tích hợp Hệ thống (Integration)

Toàn bộ model đã được nén về 4-bit (chỉ nặng **~2.0GB**, tối ưu cho máy cá nhân) và đăng ký thành công vào hệ thống.

### 4.1. Cấu hình Ollama (`modelfiles/Modelfile_llama_3.2_fable...`)

Thiết lập các tham số khống chế độ đa dạng và độ dài, tránh model sinh lan man:

```dockerfile
FROM ../../models/Llama-3.2-3B-Instruct-fable-1000.Q4_K_M.gguf
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
```

### 4.2. Khai báo API (`config/models.json`)

Bổ sung đầy đủ 4 biến thể để phục vụ tính năng **Compare Mode** trên UI:

| Tên biến thể | Mô tả |
|---|---|
| `llama-fp16` | Base model 16-bit (mốc tham chiếu gốc) |
| `llama-q4` | Base model 4-bit (mốc đánh giá hao hụt do lượng tử hóa) |
| `llama3-fable-300-q4` | Finetuned 1 epoch |
| `llama3-fable-1000-q4` | Finetuned 3 epoch |

---

## 5. Đánh giá Mô hình (Quick Evaluation)

Sau khi tích hợp vào ứng dụng, các biến thể mô hình được đánh giá bằng cơ chế **Quick Evaluation**.

- **Judge Model:** `Llama 3.2 3B FP16` (base)
- **Đối tượng đánh giá:** `llama-q4`, `llama3-fable-300-q4`, `llama3-fable-1000-q4`
- **Phương thức:** Các mô hình sinh đầu ra từ cùng một prompt, sau đó toàn bộ kết quả được chuyển đến mô hình **FP16** để thực hiện đánh giá và xếp hạng.