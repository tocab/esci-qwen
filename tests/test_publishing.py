import json

import numpy as np
import polars as pl
import pytest

from src.common import LABELS
from src.data import load_split, prepare
from src.jev import PromptEncoder
from src.predict import predict
from src.subset import make_subset
from src.validation import infer
from tests.helpers import synthetic_data


def test_subset_keeps_connected_groups_and_excludes_test(tmp_path):
    examples, products = synthetic_data()
    examples.write_parquet(tmp_path / "examples.parquet")
    products.write_parquet(tmp_path / "products.parquet")
    source = tmp_path / "prepared"
    prepare(tmp_path / "examples.parquet", tmp_path / "products.parquet", source)
    train = load_split(source, "train")
    # A connected group spans locales; all its rows must be retained together.
    train = train.with_columns(
        pl.when(pl.col("esci_label") == "E")
        .then(pl.lit("jp"))
        .otherwise(pl.col("product_locale"))
        .alias("product_locale")
    )
    train.write_parquet(source / "train.parquet")
    from src.common import file_hash, write_json

    manifest = json.loads((source / "manifest.json").read_text())
    manifest["split_sha256"]["train"] = file_hash(source / "train.parquet")
    write_json(source / "manifest.json", manifest)
    first = make_subset(source, tmp_path / "first", 1, 1)
    make_subset(source, tmp_path / "repeat", 1, 1)
    selected = load_split(tmp_path / "first", "train")
    assert selected.equals(load_split(tmp_path / "repeat", "train"))
    expected = train.filter(pl.col("query_group").is_in(selected["query_group"].unique().to_list()))
    assert selected.equals(expected)
    assert not (tmp_path / "first" / "test.parquet").exists()
    assert first["prepared_audit"]["cross_partition"]["query_id"] == 0
    assert not first["test_evaluation_permitted"]
    with pytest.raises(FileExistsError):
        make_subset(source, tmp_path / "first", 1, 1)


def test_prediction_matches_validation(model, tokenizer):
    rows = [
        dict(example_id="demo", query="wireless mouse", product_title="mouse", product_locale="us")
    ]
    encoder = PromptEncoder(tokenizer, max_length=256)
    result = list(predict(model, encoder, rows))[0]
    expected = infer(model, encoder, pl.DataFrame(rows))[0]
    assert result["example_id"] == "demo"
    assert result["label"] == LABELS[expected.argmax()]
    np.testing.assert_array_equal(list(result["probabilities"].values()), expected)
    assert list(result["probabilities"]) == list(LABELS)
    with pytest.raises(ValueError, match="nonempty query"):
        list(predict(model, encoder, [dict(query="", product_locale="us")]))
