"""Validation metrics for ESCI fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from .common import LABELS
from .jev import PromptEncoder
from .models import kernel_metadata, score_batch


def metrics(labels: Sequence[str], probabilities: ArrayLike) -> dict[str, Any]:
    """Validate N-by-4 E/S/C/I probabilities and compute classification/calibration metrics."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), 4) or not len(labels):
        raise ValueError("Expected nonempty N-by-4 probabilities")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid probability distributions")
    # Float32 softmax sums can differ by ~1e-7; sklearn checks float64 epsilon.
    probabilities = probabilities / probabilities.sum(1, keepdims=True)
    truth = np.array([LABELS.index(label) for label in labels])
    predicted = probabilities.argmax(1)
    predicted_labels = [LABELS[i] for i in predicted]
    confidence = probabilities.max(1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(
                (predicted[in_bin] == truth[in_bin]).mean() - confidence[in_bin].mean()
            )
    return {
        "rows": len(labels),
        "macro_f1": f1_score(
            labels, predicted_labels, labels=list(LABELS), average="macro", zero_division=0
        ),
        "micro_f1": f1_score(
            labels, predicted_labels, labels=list(LABELS), average="micro", zero_division=0
        ),
        "accuracy": accuracy_score(truth, predicted),
        "per_class": classification_report(
            labels, predicted_labels, labels=list(LABELS), zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(4))).tolist(),
        "log_loss": log_loss(truth, probabilities, labels=list(range(4))),
        "brier_score": float(np.square(probabilities - np.eye(4)[truth]).sum(1).mean()),
        "ece_10_equal_width": float(ece),
    }


def bootstrap_macro_f1(
    labels: Sequence[str],
    probabilities: ArrayLike,
    query_groups: ArrayLike,
    samples: int = 200,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Estimate a 95% macro-F1 interval by resampling whole connected query groups.

    Return None when samples is nonpositive, disabling bootstrap computation.
    """
    if samples <= 0:
        return None
    groups, inverse = np.unique(query_groups, return_inverse=True)
    counts = np.zeros((len(groups), 4, 4), dtype=np.int64)
    truth = np.array([LABELS.index(label) for label in labels])
    predicted = np.asarray(probabilities).argmax(1)
    np.add.at(counts, (inverse, truth, predicted), 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        matrix = counts[rng.integers(0, len(groups), len(groups))].sum(0)
        numerator = 2 * np.diag(matrix)
        denominator = matrix.sum(0) + matrix.sum(1)
        values.append(
            np.divide(numerator, denominator, out=np.zeros(4), where=denominator != 0).mean()
        )
    return {
        "method": "query-connected group percentile bootstrap",
        "samples": samples,
        "seed": seed,
        "groups": len(groups),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


@torch.inference_mode()
def infer(
    model: torch.nn.Module, encoder: PromptEncoder, frame: pl.DataFrame, batch_size: int = 4
) -> NDArray[np.float32]:
    """Score a nonempty frame in evaluation mode and return N-by-4 float32 probabilities."""
    model.eval()
    rows = frame.to_dicts()
    probabilities = []
    for start in range(0, len(rows), batch_size):
        _, p = score_batch(model, encoder, rows[start : start + batch_size])
        probabilities.append(p.cpu().numpy())
    if not probabilities:
        raise ValueError("Cannot evaluate an empty split")
    return np.concatenate(probabilities)


def evaluate_frame(
    model: torch.nn.Module,
    encoder: PromptEncoder,
    frame: pl.DataFrame,
    batch_size: int = 4,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> tuple[dict[str, Any], NDArray[np.float32]]:
    """Return aggregate/locale metrics, optional group bootstrap, runtime data, and predictions."""
    on_cuda = next(model.parameters()).device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    probabilities = infer(model, encoder, frame, batch_size)
    if on_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = metrics(frame["esci_label"].to_list(), probabilities)
    report["per_locale"] = {}
    for locale in sorted(frame["product_locale"].unique()):
        indices = np.flatnonzero(frame["product_locale"].to_numpy() == locale)
        report["per_locale"][locale] = metrics(
            frame["esci_label"].gather(indices).to_list(), probabilities[indices]
        )
    report["bootstrap"] = bootstrap_macro_f1(
        frame["esci_label"].to_list(),
        probabilities,
        frame["query_group"].to_numpy(),
        bootstrap_samples,
        seed,
    )
    report["elapsed_seconds"] = elapsed
    report["examples_per_second"] = len(frame) / elapsed
    report["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if on_cuda else None
    report["label_order"] = LABELS
    report["prompt"] = {
        "max_length": encoder.max_length,
        "order": encoder.order,
        "prefix": "Selection: [",
        "chat_template_used": False,
    }
    report["kernels"] = kernel_metadata(model)
    report["candidate_map"] = {f"[{i}]": label for i, label in enumerate(encoder.order, 1)}
    return report, probabilities
