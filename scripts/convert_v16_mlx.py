#!/usr/bin/env python3
"""Convert the V16 Hugging Face GPT-2 checkpoint to MLX-LM layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


TRANSPOSE_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def convert_key_and_tensor(key: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    """Map Hugging Face GPT-2 names and Conv1D matrices to MLX-LM names."""

    if not key.startswith("transformer."):
        raise ValueError(f"unexpected V16 source key: {key}")
    target = "model." + key.removeprefix("transformer.")
    if key.endswith(TRANSPOSE_SUFFIXES):
        value = value.T
    return target, np.ascontiguousarray(value)


def convert_checkpoint(source: Path, destination: Path, *, force: bool = False) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir() or not (source / "model.safetensors").is_file():
        raise FileNotFoundError(f"V16 Hugging Face checkpoint not found: {source}")
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"destination already exists: {destination}; pass --force to replace it"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for path in source.glob("*.json"):
        shutil.copy2(path, destination / path.name)

    converted: dict[str, np.ndarray] = {}
    with safe_open(str(source / "model.safetensors"), framework="numpy", device="cpu") as weights:
        for key in weights.keys():
            target, value = convert_key_and_tensor(key, weights.get_tensor(key))
            converted[target] = value

    temporary = destination / "model.tmp.safetensors"
    save_file(converted, str(temporary), metadata={"format": "mlx"})
    temporary.replace(destination / "model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert_checkpoint(args.source, args.output, force=args.force)
    print(f"wrote MLX checkpoint to {args.output}")


if __name__ == "__main__":
    main()
