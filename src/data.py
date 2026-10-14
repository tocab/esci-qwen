"""Official Task 2 preparation, query-connected splits, and leakage audits."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .common import LABELS, file_hash, write_json

FIELDS = (
    "query",
    "product_title",
    "product_brand",
    "product_color",
    "product_locale",
    "product_bullet_point",
    "product_description",
)


def clean(value: Any) -> str:
    """Convert a field to text, mapping None and floating-point NaN to an empty string."""
    return (
        ""
        if value is None or isinstance(value, (float, np.floating)) and np.isnan(value)
        else str(value)
    )


def normalized_query(value: Any) -> str:
    """Normalize query text with NFKC, casefolding, and punctuation/whitespace removal."""
    text = unicodedata.normalize("NFKC", clean(value)).casefold()
    return re.sub(r"[\W_]+", "", text)


def download_official(directory: str | Path, revision: str | None = None) -> dict[str, Any]:
    """Download official ESCI files at an immutable revision, or verify an existing cache."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "source.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())
        if revision and revision != source["revision"]:
            raise ValueError("Existing source revision differs; choose a fresh directory")
        for name, digest in source["sha256"].items():
            if file_hash(directory / name) != digest:
                raise ValueError(f"Source checksum mismatch: {name}")
        return source
    if revision is None:
        request = urllib.request.Request(
            "https://api.github.com/repos/amazon-science/esci-data/commits/main",
            headers={"User-Agent": "esci-jev"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            revision = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full immutable Git commit SHA")
    hashes = {}
    urls = {}
    for name in (
        "shopping_queries_dataset_examples.parquet",
        "shopping_queries_dataset_products.parquet",
    ):
        url = (
            f"https://media.githubusercontent.com/media/amazon-science/esci-data/"
            f"{revision}/shopping_queries_dataset/{name}"
        )
        target = directory / name
        temporary = target.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        pl.read_parquet_schema(temporary)  # Reject LFS pointer/error documents.
        temporary.replace(target)
        hashes[name] = file_hash(target)
        urls[name] = url
    for name in ("LICENSE", "NOTICE"):
        url = f"https://raw.githubusercontent.com/amazon-science/esci-data/{revision}/{name}"
        urllib.request.urlretrieve(url, directory / name)
        hashes[name] = file_hash(directory / name)
        urls[name] = url
    source = {
        "repository": "https://github.com/amazon-science/esci-data",
        "revision": revision,
        "license": "Apache-2.0",
        "sha256": hashes,
        "urls": urls,
    }
    write_json(manifest, source)
    return source


def join_task2(examples: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:
    """Validate Task 2 examples and join product fields by product ID and locale."""
    required = {
        "example_id",
        "query",
        "query_id",
        "product_id",
        "product_locale",
        "esci_label",
        "large_version",
        "split",
    }
    if not required <= set(examples.columns):
        raise ValueError(f"Missing example columns: {required - set(examples.columns)}")
    selected = examples.filter(pl.col("large_version") == 1)
    if selected.is_empty() or not set(selected["esci_label"].unique()) <= set(LABELS):
        raise ValueError("Empty Task 2 subset or invalid labels")
    if not set(selected["split"].unique()) <= {"train", "test"}:
        raise ValueError("Unexpected official partition")
    missing = [
        pl.col(name).is_null()
        | (pl.col(name).is_nan() if selected.schema[name].is_float() else pl.lit(False))
        for name in required
    ]
    if (
        selected.select(pl.any_horizontal(missing).any()).item()
        or selected["example_id"].is_duplicated().any()
    ):
        raise ValueError("Missing identifiers/labels or duplicate example IDs")
    if not set(selected["product_locale"].unique()) <= {"us", "es", "jp"}:
        raise ValueError("Unsupported official locale")
    joined = selected.join(
        products.with_columns(pl.lit(True).alias("_product_found")),
        on=["product_id", "product_locale"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if joined["_product_found"].null_count():
        raise ValueError("Products are missing for some (product_id, locale) keys")
    joined = joined.drop("_product_found")
    expressions = []
    for field in FIELDS:
        expression = pl.col(field) if field in joined.columns else pl.lit("")
        if field in joined.columns and joined.schema[field].is_float():
            expression = expression.fill_nan(None)
        expressions.append(expression.cast(pl.String).fill_null("").alias(field))
    joined = joined.with_columns(expressions).with_columns(
        pl.col("example_id", "query_id", "product_id").cast(pl.String)
    )
    # Normalize unique query strings once rather than invoking Python per product.
    queries = (
        joined.select("query")
        .unique(maintain_order=True)
        .with_columns(
            pl.col("query")
            .map_elements(normalized_query, return_dtype=pl.String)
            .alias("normalized_query")
        )
    )
    joined = joined.join(queries, on="query", how="left", validate="m:1", maintain_order="left")
    if joined["normalized_query"].eq("").any():
        raise ValueError("Queries must contain a letter or digit after normalization")
    return joined


def query_components(frame: pl.DataFrame) -> pl.Series:
    """Connect global query IDs and normalized text, including across locales."""
    parent = {}

    def root(key: str) -> str:
        """Find a query-component representative, compressing its parent path."""
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for qid, query in (
        frame.select("query_id", "normalized_query").unique(maintain_order=True).iter_rows()
    ):
        a, b = root("id:" + str(qid)), root("text:" + query)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = {qid: root("id:" + str(qid)) for qid in frame["query_id"].unique()}
    return frame["query_id"].replace_strict(groups).alias("query_group")


def audit(frame: pl.DataFrame, partition: str = "split") -> dict[str, Any]:
    """Count duplicate/conflicting pairs and query overlap across named partitions."""
    pairs = ["normalized_query", "product_id", "product_locale"]
    result = {
        "exact_duplicate_pairs": int(frame.select(pairs).is_duplicated().sum()),
        "conflicting_label_pairs": frame.group_by(pairs)
        .agg(pl.col("esci_label").n_unique())
        .filter(pl.col("esci_label") > 1)
        .height,
        "normalization": "NFKC + casefold + removal of punctuation/whitespace; no semantic similarity",
        "cross_partition": {},
    }
    for key, cols in {
        "query_id": ["query_id"],
        "normalized_query": ["normalized_query"],
        "query_product": pairs,
        "query_product_across_locales": ["normalized_query", "product_id"],
    }.items():
        result["cross_partition"][key] = (
            frame.group_by(cols)
            .agg(pl.col(partition).n_unique())
            .filter(pl.col(partition) > 1)
            .height
        )
    return result


def split_task2(
    frame: pl.DataFrame, fraction: float = 0.1, seed: int = 42
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, dict[str, Any]]:
    """Split connected training groups and quarantine any groups touching official test.

    Return train/validation/test frames, quarantined training rows, and the
    original official-partition leakage audit.
    """
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    frame = frame.with_columns(query_components(frame))
    original_audit = audit(frame)
    test = frame.filter(pl.col("split") == "test")
    train = frame.filter(pl.col("split") == "train")
    test_groups = test["query_group"].unique().to_list()
    overlaps_test = pl.col("query_group").is_in(test_groups)
    quarantine = train.filter(overlaps_test)
    train = train.filter(~overlaps_test)
    groups = np.array(sorted(train["query_group"].unique().to_list()))
    if len(groups) < 2:
        raise ValueError("At least two non-test training query groups are needed")
    np.random.default_rng(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    in_validation = pl.col("query_group").is_in(groups[:count].tolist())
    validation = train.filter(in_validation)
    train = train.filter(~in_validation)
    splits = {"train": train, "validation": validation, "test": test}
    splits = {
        name: part.with_columns(pl.lit(name).alias("partition")) for name, part in splits.items()
    }
    return splits, quarantine, original_audit


def statistics(frame: pl.DataFrame) -> dict[str, Any]:
    """Summarize labels, locales, query groups, and field-length quantiles."""

    def counts(column: str) -> dict[str, int]:
        """Count rows for each distinct value of the requested string column."""
        return dict(frame.group_by(column).len().iter_rows())

    locales = sorted(frame["product_locale"].unique().to_list())
    labels = sorted(frame["esci_label"].unique().to_list())
    locale_counts = {
        (label, locale): count
        for label, locale, count in frame.group_by("esci_label", "product_locale").len().iter_rows()
    }
    quantiles = (0, 0.5, 0.9, 0.99, 1)
    lengths = frame.select(pl.col(FIELDS).str.len_chars())
    return {
        "rows": frame.height,
        "classes": counts("esci_label"),
        "locales": counts("product_locale"),
        "query_ids": frame["query_id"].n_unique(),
        "query_groups": frame["query_group"].n_unique(),
        "locale_classes": {
            label: {locale: locale_counts.get((label, locale), 0) for locale in locales}
            for label in labels
        },
        "text_character_quantiles": {
            field: {q: lengths[field].quantile(q, interpolation="linear") for q in quantiles}
            for field in FIELDS
        },
    }


def prepare(
    examples_path: str | Path,
    products_path: str | Path,
    output: str | Path,
    fraction: float = 0.1,
    seed: int = 42,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write prepared splits, row IDs, quarantine data, and a checksummed audit manifest."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared dataset already exists; use a fresh directory")
    frame = join_task2(pl.read_parquet(examples_path), pl.read_parquet(products_path))
    splits, quarantine, original_audit = split_task2(frame, fraction, seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": 2,
        "subset": "large_version == 1",
        "seed": seed,
        "validation_fraction": fraction,
        "source": source,
        "raw_sha256": {"examples": file_hash(examples_path), "products": file_hash(products_path)},
        "quarantined_train_rows": len(quarantine),
        "official_audit": original_audit,
        "prepared_audit": audit(pl.concat(list(splits.values())), "partition"),
        "splits": {},
        "split_sha256": {},
    }
    for name, part in splits.items():
        target = output / f"{name}.parquet"
        part.write_parquet(target)
        manifest["split_sha256"][name] = file_hash(target)
        manifest["splits"][name] = statistics(part)
        part.select("example_id", "query_id", "product_locale", "query_group").write_csv(
            output / f"{name}_ids.csv"
        )
    quarantine.write_parquet(output / "quarantined.parquet")
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(directory: str | Path, split: str) -> pl.DataFrame:
    """Load a prepared train, validation, or test split after verifying its checksum."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}.parquet"
    if file_hash(path) != manifest["split_sha256"][split]:
        raise ValueError(f"Prepared {split} checksum differs from manifest")
    return pl.read_parquet(path)


def main() -> None:
    """Prepare official Task 2 data using command-line download and split settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--revision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = Path(args.raw)
    source = (
        download_official(raw, args.revision)
        if args.download
        else (
            json.loads((raw / "source.json").read_text())
            if (raw / "source.json").exists()
            else None
        )
    )
    manifest = prepare(
        raw / "shopping_queries_dataset_examples.parquet",
        raw / "shopping_queries_dataset_products.parquet",
        args.output,
        args.validation_fraction,
        args.seed,
        source,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
