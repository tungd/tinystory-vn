# Model Artifacts

GGUF files are intentionally not committed to Git because they are large.

Expected local model files / Ollama imports:

| Experiment | Ollama model | Modelfile in this folder |
| --- | --- | --- |
| SFT Clean 3K | `llama32-fable-clean3k:q4` | `Modelfile.llama32-clean3k` |
| Failure LoRA 300 | `llama32-fable-failure-lora:q4` | `Modelfile.llama32-failure-lora` |
| Fluency SFT v1 LoRA | `llama32-fable-fluency-sft-v1:q4` | `Modelfile.llama32-fluency-sft-v1` |

The app registry is in:

```text
config/models.json
```

Large GGUF files should be stored outside Git, for example in local Ollama storage or a shared Drive/Kaggle artifact.
