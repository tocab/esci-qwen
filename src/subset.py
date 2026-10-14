"""Select the reported experiment's query groups from prepared train/validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .common import file_hash, write_json
from .data import audit, load_split, statistics


def make_subset(
    source: str | Path,
    output: str | Path,
    train_groups_per_locale: int = 3000,
    validation_groups_per_locale: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Sample complete query groups per locale and write checksummed train/validation subsets."""
    if min(train_groups_per_locale, validation_groups_per_locale) < 1:
        raise ValueError("Group counts must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Subset output already exists; use a fresh directory")
    rng = np.random.default_rng(seed)
    parts = {}
    for split, count in [
        ("train", train_groups_per_locale),
        ("validation", validation_groups_per_locale),
    ]:
        frame = load_split(source, split)
        selected = set()
        for locale in sorted(frame["product_locale"].unique().to_list()):
            groups = np.array(
                sorted(
                    frame.filter(pl.col("product_locale") == locale)["query_group"]
                    .unique()
                    .to_list()
                )
            )
            selected.update(rng.choice(groups, min(count, len(groups)), replace=False))
        # Include whole connected groups, including their rows in other locales.
        parts[split] = frame.filter(pl.col("query_group").is_in(sorted(selected)))
    manifest = {
        "purpose": "reported experiment training/validation subset; official test excluded",
        "test_evaluation_permitted": False,
        "seed": seed,
        "parent_manifest_sha256": file_hash(Path(source) / "manifest.json"),
        "train_groups_per_locale": train_groups_per_locale,
        "validation_groups_per_locale": validation_groups_per_locale,
        "prepared_audit": audit(pl.concat(list(parts.values())), "partition"),
        "splits": {},
        "split_sha256": {},
    }
    output.mkdir(parents=True)
    for split, part in parts.items():
        target = output / f"{split}.parquet"
        part.write_parquet(target)
        manifest["split_sha256"][split] = file_hash(target)
        manifest["splits"][split] = statistics(part)
        part.select("example_id", "query_id", "product_locale", "query_group").write_csv(
            output / f"{split}_ids.csv"
        )
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    """Create an experiment subset using command-line group counts and random seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/prepared")
    parser.add_argument("--output", default="data/extended")
    parser.add_argument("--train-groups-per-locale", type=int, default=3000)
    parser.add_argument("--validation-groups-per-locale", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = make_subset(
        args.source,
        args.output,
        args.train_groups_per_locale,
        args.validation_groups_per_locale,
        args.seed,
    )
    print({split: stats["rows"] for split, stats in manifest["splits"].items()})


if __name__ == "__main__":
    main()
