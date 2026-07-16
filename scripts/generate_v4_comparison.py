"""Generate matched v3-full/v4 stories from v4-held-out controls."""

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_v3_comparison import generate_for_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", required=True)
    parser.add_argument("--v4", required=True)
    parser.add_argument("--controls-file", default="runs/v4/data/eval_controls.json")
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    controls = json.loads(Path(args.controls_file).read_text())[: args.controls]
    if not controls:
        raise ValueError("No v4 evaluation controls")
    rows = []
    rows.extend(generate_for_model("v3-full", args.v3, controls, args))
    rows.extend(generate_for_model("v4", args.v4, controls, args))
    result = {
        "kind": "v4-generation-comparison",
        "controls": {
            "source": args.controls_file,
            "count": len(controls),
            "first": controls[0]["source"],
            "last": controls[-1]["source"],
        },
        "settings": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
            "max_new_tokens": args.max_new_tokens,
            "batch": args.batch,
            "seed": args.seed,
        },
        "generations": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
