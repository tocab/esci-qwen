"""Evaluate a frozen Qwen checkpoint on a prepared ESCI split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import seed_all, write_json
from .data import load_split
from .jev import PromptEncoder
from .models import load_model
from .validation import evaluate_frame


def main() -> None:
    """Evaluate a saved checkpoint on a prepared split and write its metrics report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Evaluation output already exists; use a fresh file")
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    seed_all(config["training"]["seed"])
    frame = load_split(args.data, args.split)
    model, tokenizer, _ = load_model(config, args.checkpoint)
    encoder = PromptEncoder(tokenizer, **config["prompt"])
    report, _ = evaluate_frame(
        model, encoder, frame, **config["evaluation"], seed=config["training"]["seed"]
    )
    report["split"] = args.split
    write_json(args.output, report)
    print({key: report[key] for key in ("rows", "macro_f1", "accuracy")})


if __name__ == "__main__":
    main()
