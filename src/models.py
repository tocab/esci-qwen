"""Qwen loading and language-backbone LoRA adapters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .jev import PromptEncoder, candidate_scores, label_probabilities


@contextmanager
def native_qwen_kernels() -> Iterator[None]:
    """Construct CPU models safely even when CUDA/FLA are installed on the host.

    Transformers chooses FLA at module import based on CUDA availability rather
    than the eventual model device. Keep this construction context synchronous.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    names = (
        "FusedRMSNormGated",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    saved = {name: getattr(implementation, name) for name in names}
    try:
        for name in names:
            setattr(implementation, name, None)
        yield
    finally:
        for name, value in saved.items():
            setattr(implementation, name, value)


def kernel_metadata(model: nn.Module) -> dict[str, list[str]]:
    """Identify the DeltaNet rule implementations and gated norms selected by the model."""
    rules = set()
    norms = set()
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            rules.add(module.chunk_gated_delta_rule.__module__)
            norms.add(type(module.norm).__module__ + "." + type(module.norm).__name__)
    return {"delta_rule_implementations": sorted(rules), "gated_norms": sorted(norms)}


def add_lora(backbone: PreTrainedModel, training: Mapping[str, Any]) -> tuple[PeftModel, list[str]]:
    """Attach LoRA to actual language-backbone linear paths and return their names."""
    # Resolve actual Qwen module paths, including DeltaNet in_proj_* layers.
    targets = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and ".language_model." in name and "lm_head" not in name
    ]
    if not targets:
        raise ValueError("No Qwen language-model linear layers found; inspect architecture")
    model = get_peft_model(
        backbone,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=training["lora_r"],
            lora_alpha=training["lora_alpha"],
            lora_dropout=training["lora_dropout"],
            target_modules=targets,
            bias="none",
        ),
    )
    return model, targets


def load_model(
    config: Mapping[str, Any], checkpoint: str | Path | None = None, trainable: bool = False
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase, list[str]]:
    """Load pinned Qwen/tokenizer weights and optionally create or restore LoRA adapters.

    Select native kernels for CPU, validate CUDA/bf16 availability, and enable
    gradient checkpointing only when preparing a trainable model.
    """
    spec = config["model"]
    if spec["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Repair the NVIDIA driver or choose CPU/float32.")
    if spec["dtype"] == "bfloat16" and spec["device"].startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 is not supported on this GPU")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[spec["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["name"], revision=spec["revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backend = spec.get("kernel_backend", "auto")
    if backend not in ("auto", "native"):
        raise ValueError("kernel_backend must be auto or native")
    native = spec["device"] == "cpu" or backend == "native"
    with native_qwen_kernels() if native else nullcontext():
        backbone = AutoModelForImageTextToText.from_pretrained(
            spec["name"],
            revision=spec["revision"],
            dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
    backbone.config.use_cache = False
    targets = []
    if checkpoint:
        backbone = PeftModel.from_pretrained(
            backbone, Path(checkpoint) / "adapter", is_trainable=trainable
        )
    elif trainable:
        backbone, targets = add_lora(backbone, config["training"])
    else:
        backbone.requires_grad_(False)
    if trainable and config["training"]["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        backbone.enable_input_require_grads()
    backbone.to(spec["device"])
    return backbone, tokenizer, targets


def score_batch(
    model: nn.Module, encoder: PromptEncoder, records: Sequence[Mapping[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return B-by-4 candidate log scores and probabilities in fixed E/S/C/I order."""
    prompts = [encoder.encode(record) for record in records]
    scores = candidate_scores(model, prompts, encoder.candidates, encoder.tokenizer.pad_token_id)
    probabilities = label_probabilities(scores, encoder.order)
    return scores, probabilities


def save_weights(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, directory: str | Path
) -> None:
    """Save adapter-only weights and tokenizer files under a checkpoint directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter", save_embedding_layers=False)
    tokenizer.save_pretrained(directory / "tokenizer")
