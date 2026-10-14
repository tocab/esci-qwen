import numpy as np
import polars as pl
import pytest

from src.data import (
    audit,
    clean,
    join_task2,
    load_split,
    normalized_query,
    prepare,
    split_task2,
    statistics,
)
from tests.helpers import synthetic_data


def test_join_uses_locale_and_nulls():
    examples, products = synthetic_data()
    product = products.head(1).with_columns(
        pl.lit("es").alias("product_locale"), pl.lit("different locale").alias("product_title")
    )
    products = pl.concat([products, product])
    joined = join_task2(examples, products)
    assert joined["product_title"][0] == "apple"
    assert joined["product_brand"][0] == ""
    assert joined["example_id"].to_list() == examples["example_id"].to_list()
    examples = examples.with_columns(
        pl.when(pl.col("example_id") == examples["example_id"][0])
        .then(0)
        .otherwise(pl.col("large_version"))
        .alias("large_version")
    )
    assert len(join_task2(examples, products)) == len(examples) - 1
    with pytest.raises(ValueError, match="Products are missing"):
        join_task2(examples, products.head(1))
    with pytest.raises(pl.exceptions.ComputeError):
        join_task2(examples, pl.concat([products, products.head(1)]))


def test_group_split_quarantines_test_and_connects_text():
    examples, products = synthetic_data()
    examples = examples.with_columns(
        pl.when(pl.col("query_id") == 1)
        .then(pl.lit("FRUIT 0!"))
        .when(pl.col("query_id") == 2)
        .then(pl.lit("fruit 8"))
        .otherwise(pl.col("query"))
        .alias("query")
    )
    joined = join_task2(examples, products)
    splits, quarantine, report = split_task2(joined, 0.25, 42)
    assert len(quarantine) == 4
    assert report["cross_partition"]["normalized_query"] == 1
    assert len(splits["test"]) == 4
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(splits[a]["query_id"]).isdisjoint(splits[b]["query_id"])
        assert set(splits[a]["normalized_query"]).isdisjoint(splits[b]["normalized_query"])
    repeat, _, _ = split_task2(joined, 0.25, 42)
    assert (
        splits["validation"]["example_id"].to_list() == repeat["validation"]["example_id"].to_list()
    )
    assert audit(pl.concat(list(splits.values())), "partition")["cross_partition"] == {
        "query_id": 0,
        "normalized_query": 0,
        "query_product": 0,
        "query_product_across_locales": 0,
    }


def test_manifests_and_checksum(tmp_path):
    examples, products = synthetic_data()
    examples.write_parquet(tmp_path / "examples.parquet")
    products.write_parquet(tmp_path / "products.parquet")
    output = tmp_path / "prepared"
    manifest = prepare(tmp_path / "examples.parquet", tmp_path / "products.parquet", output)
    assert sum(s["rows"] for s in manifest["splits"].values()) == 36
    frame = load_split(output, "train")
    assert isinstance(frame, pl.DataFrame)
    assert len(frame) > 0
    assert (
        pl.read_csv(output / "train_ids.csv")["example_id"].to_list()
        == frame["example_id"].to_list()
    )
    (output / "train.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_split(output, "train")


def test_normalization():
    assert normalized_query("ＭＯＵＳＥ - USB") == normalized_query("mouse usb")
    assert normalized_query("マウス") == "マウス"
    assert clean(None) == clean(float("nan")) == clean(np.float32("nan")) == ""
    assert clean("nan") == "nan"


def test_audit_counts_all_duplicate_rows_and_conflicting_pairs():
    examples, products = synthetic_data()
    frame = join_task2(examples, products)
    duplicate = frame.head(1).with_columns(pl.lit("S").alias("esci_label"))
    report = audit(pl.concat([frame, duplicate]))
    assert report["exact_duplicate_pairs"] == 2
    assert report["conflicting_label_pairs"] == 1


def test_statistics_unicode_quantiles_and_zero_counts():
    examples, products = synthetic_data()
    frame = split_task2(join_task2(examples, products))[0]["train"]
    frame = frame.with_columns(pl.lit("マウス").alias("product_title"))
    report = statistics(frame)
    assert report["text_character_quantiles"]["product_title"] == {
        0: 3,
        0.5: 3,
        0.9: 3,
        0.99: 3,
        1: 3,
    }
    sparse = frame.filter((pl.col("product_locale") == "us") | (pl.col("esci_label") == "E"))
    assert statistics(sparse)["locale_classes"]["S"]["es"] == 0


def test_missing_ids_and_nonempty_normalized_queries():
    examples, products = synthetic_data()
    invalid = examples.with_columns(pl.lit(None).cast(pl.Int64).alias("query_id"))
    with pytest.raises(ValueError, match="Missing identifiers"):
        join_task2(invalid, products)
    invalid = examples.with_columns(pl.lit("___ !").alias("query"))
    with pytest.raises(ValueError, match="after normalization"):
        join_task2(invalid, products)
    products = products.with_columns(pl.lit(float("nan")).alias("product_brand"))
    assert join_task2(examples, products)["product_brand"].unique().to_list() == [""]
