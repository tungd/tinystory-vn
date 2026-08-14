# TinyStory Demo Studio

This is a separate operator-facing app. It does not replace or modify the
existing Ollama-oriented generator under `app/` and `web/`.

It exposes the five representative systems from the global benchmark:

| Direction | System | Runtime |
| --- | --- | --- |
| E1 | SLM 60M | llama.cpp completion |
| E2 | V16 conditioned | MLX-LM |
| E3 | SmolLM2 135M + LoRA C | Transformers + PEFT |
| E4 | Base + Repair | llama.cpp plus validator/self-repair |
| E5 | Llama 3.2 3B fable Q4 | llama.cpp chat |

E2 uses the selected V16 conditioning checkpoint in
`runs/v16/artifacts/conditioning-mlx` and keeps its original inference contract
(`<char>`, `<moral>`, then `<story>`); the other fields remain in the shared
brief for judging and for the other systems. The judge evaluates the requested
narrative elements, while the literal `Moral:` format is reported separately.

Each row has an explicit Load/Unload action. A loaded row can run the shared
brief in the center canvas. Select two or more loaded rows with Compare, then
run the same brief across them and judge each output side by side. The Case
selector exposes the canonical `GJ01`–`GJ25` global-judge briefs from
`results/global_judge/global_prompts_v1.jsonl`; editing a field switches back to
a custom brief. The Judge panel calls the configured Google GenAI judge
(`FABLE_JUDGE_MODEL_ID` / `FABLE_JUDGE_API_KEY`) independently of every local
model runtime; it never resolves a judge through Ollama. The judge uses the
compact four-integer schema from the final benchmark protocol.

Run it from the repository root with the project environment:

```bash
.venv/bin/python -m demo.server
```

Then open <http://127.0.0.1:8010>. Runtime logs are written to `demo/logs/`.
