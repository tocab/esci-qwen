import json

import torch
from peft import PeftModel

from src.common import load_config
from src.data import prepare
from src.models import add_lora
from tests.helpers import synthetic_data, tiny_model, tiny_tokenizer
from src.train import train


import pytest


@pytest.mark.parametrize("periodic", [False, True])
def test_checkpoint_resume(tmp_path, monkeypatch, periodic):
    import src.train as training

    examples, products = synthetic_data()
    examples.write_parquet(tmp_path / "examples.parquet")
    products.write_parquet(tmp_path / "products.parquet")
    prepare(tmp_path / "examples.parquet", tmp_path / "products.parquet", tmp_path / "data")
    config = load_config("configs/train.yaml")
    config["model"].update(device="cpu", dtype="float32")
    config["data"]["directory"] = str(tmp_path / "data")
    config["prompt"]["max_length"] = 256
    config["training"].update(
        epochs=2, batch_size=6, accumulation_steps=2, gradient_checkpointing=False, lora_dropout=0.1
    )
    if periodic:
        config["training"].update(
            validation_interval_seconds=0,
            checkpoint_interval_seconds=0,
            retain_validation_checkpoints=True,
            patience=20,
        )

    def loader(config, checkpoint=None, trainable=False):
        tokenizer = tiny_tokenizer()
        backbone = tiny_model(len(tokenizer))
        if checkpoint:
            backbone = PeftModel.from_pretrained(
                backbone, checkpoint / "adapter", is_trainable=True
            )
            targets = []
        else:
            backbone, targets = add_lora(backbone, config["training"])
        return backbone, tokenizer, targets

    monkeypatch.setattr(training, "load_model", loader)
    complete = train(config, tmp_path / "complete")
    assert complete["next_epoch"] == 2
    assert (tmp_path / "complete" / "best" / "adapter" / "adapter_model.safetensors").exists()
    # Simulate interruption after an epoch is safely saved.
    evaluate = training.evaluate_frame
    calls = 0

    def interrupt_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate_frame", interrupt_on_second)
    with pytest.raises(RuntimeError, match="simulated"):
        train(config, tmp_path / "interrupted")
    saved = torch.load(tmp_path / "interrupted" / "last" / "training_state.pt", weights_only=False)
    assert "pending_validation" in saved["state"]
    if periodic:
        assert 0 < saved["state"]["next_batch"] < 5
    monkeypatch.setattr(training, "evaluate_frame", evaluate)
    resumed = train(config, tmp_path / "interrupted", resume=True)
    assert resumed["history"] == complete["history"]
    assert (
        json.loads((tmp_path / "interrupted" / "result.json").read_text())["test_evaluated"]
        is False
    )


def test_deadline_saves_resumable_state(tmp_path, monkeypatch):
    import src.train as training

    examples, products = synthetic_data()
    examples.write_parquet(tmp_path / "examples.parquet")
    products.write_parquet(tmp_path / "products.parquet")
    prepare(tmp_path / "examples.parquet", tmp_path / "products.parquet", tmp_path / "data")
    config = load_config("configs/train.yaml")
    config["model"].update(device="cpu", dtype="float32")
    config["data"]["directory"] = str(tmp_path / "data")
    config["training"].update(
        deadline_utc="2000-01-01T00:00:00+00:00", gradient_checkpointing=False
    )

    def loader(config, *args, **kwargs):
        tokenizer = tiny_tokenizer()
        backbone, targets = add_lora(tiny_model(len(tokenizer)), config["training"])
        return backbone, tokenizer, targets

    monkeypatch.setattr(training, "load_model", loader)
    result = train(config, tmp_path / "deadline")
    assert result["stop_reason"] == "deadline_reached"
    assert result["global_update"] == 0
    assert (tmp_path / "deadline" / "last" / "training_state.pt").exists()


@pytest.mark.parametrize(
    "score,grace,expected",
    [
        (0.2, 0, "validation_plateau"),
        (0.2, 1e9, "epochs_completed"),
        (0.6, 1e9, "validation_target_reached"),
    ],
)
def test_validation_stopping_policy(tmp_path, monkeypatch, score, grace, expected):
    import src.train as training

    examples, products = synthetic_data()
    examples.write_parquet(tmp_path / "examples.parquet")
    products.write_parquet(tmp_path / "products.parquet")
    prepare(tmp_path / "examples.parquet", tmp_path / "products.parquet", tmp_path / "data")
    config = load_config("configs/train.yaml")
    config["model"].update(device="cpu", dtype="float32")
    config["data"]["directory"] = str(tmp_path / "data")
    config["training"].update(
        epochs=1,
        batch_size=4,
        accumulation_steps=1,
        gradient_checkpointing=False,
        validation_interval_seconds=0,
        min_delta=0.005,
        patience=2,
        early_stop_min_training_seconds=grace,
        stop_at_validation_macro_f1=0.5748,
    )

    def loader(config, *args, **kwargs):
        tokenizer = tiny_tokenizer()
        backbone, targets = add_lora(tiny_model(len(tokenizer)), config["training"])
        return backbone, tokenizer, targets

    monkeypatch.setattr(training, "load_model", loader)
    monkeypatch.setattr(
        training, "evaluate_frame", lambda *args, **kwargs: ({"macro_f1": score}, None)
    )
    result = train(config, tmp_path / "run")
    assert result["stop_reason"] == expected
    assert (tmp_path / "run" / "best" / "training_state.pt").exists()
