# Colab Training Runbook — fable-200M

Train a ~200M fable transformer **from scratch** on TF1-EN-3M, conditioned on
`(character, moral)` keyword seeds. Orchestrated from the terminal with
`google-colab-cli` (no browser needed).

## 0. Install the CLI (uv)

```bash
uv tool install google-colab-cli
colab --help
```

## 1. Train (Notebook A)

```bash
colab new -s trainer --gpu T4

# upload the prep + metrics scripts the notebooks import
colab upload -s trainer scripts/prepare_tf1.py      /content/scripts/prepare_tf1.py
colab upload -s trainer scripts/fable_tokenizer.py  /content/scripts/fable_tokenizer.py
colab upload -s trainer scripts/metrics.py         /content/scripts/metrics.py

# run the training notebook (streams TF1, trains BPE, trains GPT2, saves to Drive)
colab exec -s trainer -f notebooks/train_fable200m_colab.ipynb

# pull the checkpoint back
colab download -s trainer /content/drive/MyDrive/fable200m ./models/

colab stop -s trainer
```

`train_fable200m_colab.ipynb` uses `transformers` (`GPT2LMHeadModel`, `Trainer`) and
`tokenizers`. The notebook currently uses a **pipeline-validation default** of
`N_FABLES = 20_000`, `max_steps = 300` (~5 min on an L4) to verify the whole
train→save path before a real run. Scale up for the actual model: set
`N_FABLES = 200_000` (or more) and raise `max_steps` in the notebook, then provision a
larger GPU (`--gpu L4|A100`). Architecture target: BPE vocab 8192, ~200M params
(`n_embd=1024, n_layer=16, n_head=16`), bf16.

## 2. Export for the local app (optional)

```bash
# convert HF checkpoint -> GGUF q8 (needs llama.cpp)
python convert_hf_to_gguf.py models/fable200m --outfile models/fable200m-q8.gguf --q8_0

# create an Ollama model from the GGUF
cat > /tmp/Modelfile.fable200m <<'EOF'
FROM ./models/fable200m-q8.gguf
PARAMETER temperature 0.9
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
PARAMETER num_ctx 1024
EOF
ollama create fable-200m -f /tmp/Modelfile.fable200m
```

The current app path uses the converted MLX model instead. Install it with
`uv sync --extra dev --extra inference`, then run
`uv run python -m mlx_lm server --model models/fable-64m-mlx --port 8080`, then
set `FABLE_BACKEND=openai` and `OLLAMA_BASE_URL=http://127.0.0.1:8080`.
`config/models.json` already maps `fable-200m` to `fable-64m-mlx`.

## 3. Eval + Generate (Notebook B)

```bash
colab new -s eval --gpu T4
colab upload -s eval scripts/metrics.py /content/scripts/metrics.py
colab exec -s eval -f notebooks/eval_gen_fable200m_colab.ipynb
colab download -s eval /content/drive/MyDrive/fable200m/eval_summary.json ./results/
colab stop -s eval
```

Notebook B generates fables from seed prompts, computes reference-free metrics
(Distinct-1/2, Self-BLEU, Flesch) and a 4-axis LLM-as-judge (small HF judge model),
writing `results/eval_summary.json` in the shape the Results tab expects.

## Notes

- `--keep` can be passed to `colab new`/`colab run` to retain the VM between steps
  instead of upload/download each time.
- For larger subsets or faster epochs, switch to a paid GPU (`--gpu L4|A100`).
