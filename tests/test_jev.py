import math

import pytest
import torch

from src.common import LABELS
from src.jev import (
    PromptEncoder,
    candidate_loss,
    candidate_scores,
    label_probabilities,
    normalized_probabilities,
    sequence_log_scores,
)
from src.models import add_lora, score_batch


def state(**kwargs):
    return dict(query="fruit", product_title="apple", product_locale="us", **kwargs)


def test_hand_calculated_multitoken():
    logits = torch.tensor(
        [
            [[math.log(0.2), math.log(0.8)], [math.log(0.6), math.log(0.4)]],
            [[math.log(0.3), math.log(0.7)], [math.log(0.9), math.log(0.1)]],
        ]
    )
    targets = torch.tensor([[1, 0], [0, 1]])
    mask = torch.tensor([[True, True], [True, True]])
    scores = sequence_log_scores(logits, targets, mask)
    torch.testing.assert_close(scores, torch.log(torch.tensor([0.48, 0.03])))
    p = normalized_probabilities(scores)
    torch.testing.assert_close(p, torch.tensor([0.48 / 0.51, 0.03 / 0.51]))
    loss = candidate_loss(scores[None], torch.tensor([0]))
    assert loss.item() == pytest.approx(-math.log(0.48 / 0.51))
    mask[1, 1] = False
    assert sequence_log_scores(logits, targets, mask)[1].item() == pytest.approx(math.log(0.3))


def test_scores_use_full_autoregressive_suffix(model, tokenizer):
    prompt = [2, 3, 4]
    candidates = [[5, 6], [7], [8, 9, 10], [11, 12]]
    scores = candidate_scores(model, [prompt], candidates, tokenizer.pad_token_id)
    reference = []
    for candidate in candidates:
        ids = torch.tensor([prompt + candidate])
        logits = model(input_ids=ids, use_cache=False).logits
        value = sum(
            torch.log_softmax(logits[0, len(prompt) - 1 + t].float(), -1)[token]
            for t, token in enumerate(candidate)
        )
        reference.append(value)
    torch.testing.assert_close(scores[0], torch.stack(reference), atol=2e-5, rtol=2e-5)


def test_batching_and_missing_fields(model, tokenizer):
    encoder = PromptEncoder(tokenizer, 256)
    rows = [state(), state(product_description="orange " * 1000)]
    _, p = score_batch(model, encoder, rows)
    _, single = score_batch(model, encoder, rows[:1])
    torch.testing.assert_close(p[:1], single, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.sum(1), torch.ones(2))
    assert max(len(encoder.encode(row)) for row in rows) <= 254


def test_option_order_mapping(tokenizer):
    encoder = PromptEncoder(tokenizer, 256, ["I", "C", "E", "S"])
    assert encoder.order.index("E") == 2
    p = label_probabilities(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), encoder.order)
    torch.testing.assert_close(p, torch.softmax(torch.tensor([[3.0, 4.0, 2.0, 1.0]]), -1))
    assert encoder.audit([state()])["candidate_map"]["[3]"] == "E"
    assert all(len(candidate) == 2 for candidate in encoder.candidates)
    with pytest.raises(ValueError):
        PromptEncoder(tokenizer, order=["E", "E", "C", "I"])


def test_lora_gradients_on_hybrid_layers(model, tokenizer):
    model, targets = add_lora(model, dict(lora_r=4, lora_alpha=8, lora_dropout=0))
    scores, _ = score_batch(model, PromptEncoder(tokenizer, 256), [state()])
    candidate_loss(scores, torch.tensor([LABELS.index("C")])).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
        if "lora_" in name
    )
    assert any("linear_attn" in target for target in targets)
    assert not any("visual" in target for target in targets)


def test_invalid_inputs_and_nonfinite(tokenizer):
    encoder = PromptEncoder(tokenizer, 256)
    with pytest.raises(ValueError, match="query"):
        encoder.encode({"query": None, "product_locale": "us"})
    with pytest.raises(ValueError, match="locale"):
        encoder.encode({"query": "fruit", "product_locale": "de"})
    with pytest.raises(ValueError, match="too small"):
        PromptEncoder(tokenizer, 8).encode(state())
    with pytest.raises(ValueError, match="Nonfinite"):
        normalized_probabilities(torch.tensor([[float("nan"), 0.0, 1.0, 2.0]]))


def test_rejects_prefix_collisions_and_boundary_merges(tokenizer):
    class CollisionTokenizer:
        def encode(self, text, **kwargs):
            return {"1]": [1], "2]": [1, 2], "3]": [3], "4]": [4]}[text]

    with pytest.raises(ValueError, match="prefix-free"):
        PromptEncoder(CollisionTokenizer())

    class BoundaryTokenizer:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def encode(self, text, **kwargs):
            ids = tokenizer.encode(text, **kwargs)
            if text.endswith("Selection: [1]"):
                ids[-3] = 0  # Simulate a BPE token merging across the answer boundary.
            return ids

    with pytest.raises(ValueError, match="answer boundary"):
        PromptEncoder(BoundaryTokenizer(), 256).encode(state())
