"""Single-device LoRA training with resumable update-boundary checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

from .common import environment, file_hash, load_config, seed_all, write_json
from .data import load_split
from .jev import PromptEncoder, candidate_loss
from .models import kernel_metadata, load_model, save_weights, score_batch
from .validation import evaluate_frame


def save_checkpoint(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    destination: Path,
) -> None:
    """Replace a checkpoint with adapters, configuration, optimizer state, and RNG state."""
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_weights(model, tokenizer, temporary)
    write_json(temporary / "config.json", config)
    write_json(temporary / "run_manifest.json", manifest)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        temporary / "training_state.pt",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def train(config: dict[str, Any], output: str | Path, resume: bool = False) -> dict[str, Any]:
    """Run candidate-loss LoRA training and return its final resumable state.

    Resume only with matching configuration and data manifests; periodically
    save and validate, retaining the best checkpoint and applying stop rules.
    """
    config = copy.deepcopy(config)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Run directory exists; choose a new run or --resume")
    if resume and not (output / "last" / "training_state.pt").exists():
        raise FileNotFoundError("No checkpoint to resume")
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(config["data"]["directory"])
    data_hash = file_hash(directory / "manifest.json")
    if resume:
        previous_config = json.loads((output / "last" / "config.json").read_text())
        if previous_config != config:
            raise ValueError("Resume configuration must match the saved configuration exactly")
        manifest = json.loads((output / "last" / "run_manifest.json").read_text())
        if data_hash != manifest["data_manifest_sha256"]:
            raise ValueError("Data manifest has changed since checkpoint")
    seed_all(config["training"]["seed"])
    train_frame = load_split(directory, "train")
    validation = load_split(directory, "validation")
    records = train_frame.to_dicts()
    settings = config["training"]
    if settings.get("mode", "jev") != "jev":
        raise ValueError("Only Qwen candidate-loss LoRA fine-tuning is supported")
    for key in ("batch_size", "accumulation_steps", "epochs", "patience"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    model, tokenizer, targets = load_model(
        config, output / "last" if resume else None, trainable=True
    )
    encoder = PromptEncoder(tokenizer, **config["prompt"])
    current_environment = environment()
    if not resume:
        manifest = {
            "config": config,
            "data_manifest_sha256": data_hash,
            "environment": current_environment,
            "lora_targets": targets,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "objective": "candidate-normalized complete suffix cross entropy",
        }
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "tokenizer_audit.json", encoder.audit(records[:256]))
    manifest.setdefault("execution_sessions", []).append(
        {"resumed": resume, "environment": current_environment, "kernels": kernel_metadata(model)}
    )
    write_json(output / "run_manifest.json", manifest)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    batches = math.ceil(len(records) / settings["batch_size"])
    updates_per_epoch = math.ceil(batches / settings["accumulation_steps"])
    total_updates = updates_per_epoch * settings["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, round(total_updates * settings["warmup_fraction"]), total_updates
    )
    state = {
        "next_epoch": 0,
        "best_macro_f1": -1.0,
        "bad_epochs": 0,
        "history": [],
        "next_batch": 0,
        "epoch_loss": 0.0,
        "global_update": 0,
        "training_seconds_total": 0.0,
        "seconds_since_validation": 0.0,
    }
    if resume:
        # Trusted locally-created training checkpoint contains RNG tuples.
        saved = torch.load(
            output / "last" / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if state["bad_epochs"] >= settings["patience"]:
        return state
    validation_interval = settings.get("validation_interval_seconds", float("inf"))
    checkpoint_interval = settings.get("checkpoint_interval_seconds", float("inf"))
    deadline = settings.get("deadline_utc")
    deadline = datetime.fromisoformat(deadline).timestamp() if deadline else float("inf")
    last_checkpoint = time.monotonic()

    def finish(reason: str) -> dict[str, Any]:
        """Persist the final checkpoint and result with the reason training stopped."""
        state["stop_reason"] = reason
        save_checkpoint(
            model, tokenizer, optimizer, scheduler, config, manifest, state, output / "last"
        )
        write_json(
            output / "result.json",
            {
                "best_validation_macro_f1": state["best_macro_f1"],
                "test_evaluated": False,
                "state": state,
            },
        )
        return state

    def validate(epoch: int, epoch_complete: bool) -> str | None:
        """Evaluate a saved update, record improvements, and return any stopping reason."""
        nonlocal last_checkpoint
        state["pending_validation"] = {"epoch": epoch, "epoch_complete": epoch_complete}
        # Save before inference: validation failure must not lose training work.
        save_checkpoint(
            model, tokenizer, optimizer, scheduler, config, manifest, state, output / "last"
        )
        report, _ = evaluate_frame(
            model, encoder, validation, **config["evaluation"], seed=settings["seed"]
        )
        score = report["macro_f1"]
        previous_best = state["best_macro_f1"]
        meaningful = score > state.get("patience_best", previous_best) + settings.get(
            "min_delta", 0.0
        )
        state["bad_epochs"] = 0 if meaningful else state["bad_epochs"] + 1
        if meaningful:
            state["patience_best"] = score
        improved = score > previous_best
        if improved:
            state["best_macro_f1"] = score
        event = {
            "epoch": epoch + 1,
            "train_loss": state["epoch_loss"]
            / max(1, min(len(records), state["next_batch"] * settings["batch_size"]))
            if not epoch_complete
            else state["completed_loss"],
            "validation_macro_f1": score,
            "global_update": state["global_update"],
            "epoch_complete": epoch_complete,
        }
        state["history"].append(event)
        state.pop("pending_validation", None)
        state["seconds_since_validation"] = 0.0
        report["global_update"] = state["global_update"]
        report["training_seconds_total"] = state["training_seconds_total"]
        report["evaluated_utc"] = datetime.now(timezone.utc).isoformat()
        report["training_examples_per_second"] = (
            state["global_update"]
            * settings["batch_size"]
            * settings["accumulation_steps"]
            / max(1e-9, state["training_seconds_total"])
        )
        name = (
            f"validation_epoch_{epoch + 1}"
            if epoch_complete
            else f"validation_update_{state['global_update']}"
        )
        write_json(output / f"{name}.json", report)
        if improved:
            save_checkpoint(
                model, tokenizer, optimizer, scheduler, config, manifest, state, output / "best"
            )
        if settings.get("retain_validation_checkpoints", False):
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                config,
                manifest,
                state,
                output / "checkpoints" / f"update_{state['global_update']}",
            )
        save_checkpoint(
            model, tokenizer, optimizer, scheduler, config, manifest, state, output / "last"
        )
        write_json(output / "history.json", state["history"])
        print(json.dumps(event), flush=True)
        last_checkpoint = time.monotonic()
        model.train()
        grace = settings.get("early_stop_min_training_seconds", 0.0)
        if state["training_seconds_total"] >= grace and state["bad_epochs"] >= settings["patience"]:
            return "validation_plateau"
        target = settings.get("stop_at_validation_macro_f1")
        if target is not None and score >= target:
            return "validation_target_reached"
        return None

    (output / "checkpoints").mkdir(exist_ok=True)
    if state.get("pending_validation"):
        pending = state["pending_validation"]
        reason = validate(pending["epoch"], pending["epoch_complete"])
        if reason:
            return finish(reason)
    for epoch in range(state["next_epoch"], settings["epochs"]):
        on_cuda = next(model.parameters()).device.type == "cuda"
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        model.train()
        indices = np.random.default_rng(settings["seed"] + epoch).permutation(len(records))
        optimizer.zero_grad(set_to_none=True)
        state.setdefault("next_batch", 0)
        state.setdefault("epoch_loss", 0.0)
        for update_start in range(state["next_batch"], batches, settings["accumulation_steps"]):
            if time.time() >= deadline:
                return finish("deadline_reached")
            update_started = time.monotonic()
            end_batch = min(batches, update_start + settings["accumulation_steps"])
            # Weight by examples, including a short final batch/accumulation group.
            total_examples = (
                min(len(records), end_batch * settings["batch_size"])
                - update_start * settings["batch_size"]
            )
            for batch_index in range(update_start, end_batch):
                selected = indices[
                    batch_index * settings["batch_size"] : (batch_index + 1)
                    * settings["batch_size"]
                ]
                rows = [records[int(i)] for i in selected]
                scores, _ = score_batch(model, encoder, rows)
                order = encoder.order
                labels = torch.tensor(
                    [order.index(row["esci_label"]) for row in rows], device=scores.device
                )
                loss = candidate_loss(scores, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; checkpoint not advanced")
                (loss * len(rows) / total_examples).backward()
                state["epoch_loss"] += float(loss.detach()) * len(rows)
            if epoch == 0 and update_start == 0:
                lora_grads = [
                    p.grad
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                ]
                if not lora_grads or not any(torch.count_nonzero(g).item() for g in lora_grads):
                    raise RuntimeError("No nonzero LoRA gradient reached adapters")
            torch.nn.utils.clip_grad_norm_(
                parameters, settings["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - update_started
            state["global_update"] = state.get("global_update", 0) + 1
            state["training_seconds_total"] = state.get("training_seconds_total", 0.0) + elapsed
            state["seconds_since_validation"] = state.get("seconds_since_validation", 0.0) + elapsed
            state["next_batch"] = end_batch
            update = update_start // settings["accumulation_steps"] + 1
            if update == 1 or update % 10 == 0 or update == updates_per_epoch:
                progress = {
                    "epoch": epoch + 1,
                    "update": update,
                    "updates_per_epoch": updates_per_epoch,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "latest_batch_loss": float(loss.detach()),
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }
                with (output / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
            # Epoch-end validation below also handles its state transition.
            if end_batch < batches and state["seconds_since_validation"] >= validation_interval:
                reason = validate(epoch, False)
                if reason:
                    return finish(reason)
            elif (
                state["global_update"] == 1 and math.isfinite(checkpoint_interval)
            ) or time.monotonic() - last_checkpoint >= checkpoint_interval:
                save_checkpoint(
                    model, tokenizer, optimizer, scheduler, config, manifest, state, output / "last"
                )
                last_checkpoint = time.monotonic()
        state["completed_loss"] = state["epoch_loss"] / len(records)
        state["next_epoch"] = epoch + 1
        state["next_batch"] = 0
        state["epoch_loss"] = 0.0
        reason = validate(epoch, True)
        if reason:
            return finish(reason)
    return finish("epochs_completed")


def main() -> None:
    """Start or resume a training run from command-line configuration and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args.output, args.resume)


if __name__ == "__main__":
    main()
