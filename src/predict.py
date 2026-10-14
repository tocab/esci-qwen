"""Score JSONL query/product records using a saved Qwen LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .common import LABELS, seed_all
from .jev import PromptEncoder
from .models import load_model, score_batch


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    encoder: PromptEncoder,
    records: Sequence[Mapping[str, Any]],
    batch_size: int = 4,
) -> Iterator[dict[str, Any]]:
    """Yield labels and E/S/C/I probabilities, preserving optional example IDs."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        _, probabilities = score_batch(model, encoder, batch)
        for record, values in zip(batch, probabilities.cpu().tolist()):
            result = {
                "label": LABELS[max(range(4), key=values.__getitem__)],
                "probabilities": dict(zip(LABELS, values)),
            }
            if "example_id" in record:
                result["example_id"] = record["example_id"]
            yield result


def main() -> None:
    """Score a JSONL file with a saved checkpoint and write predictions to a new file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="UTF-8 JSONL file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if target.exists():
        raise FileExistsError("Prediction output already exists; use a fresh file")
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise ValueError("Input must contain at least one JSON record")
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    if args.device:
        config["model"]["device"] = args.device
        if args.device == "cpu":
            config["model"]["dtype"] = "float32"
    seed_all(config["training"]["seed"])
    model, tokenizer, _ = load_model(config, args.checkpoint)
    encoder = PromptEncoder(tokenizer, **config["prompt"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        for result in predict(model, encoder, records, args.batch_size):
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
