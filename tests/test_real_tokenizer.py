"""Optional local check; never downloads files during tests."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.jev import PromptEncoder


TOKENIZER_PATH = Path(
    ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/"
    "f75361505deaead5ce6a4a25bd2f433cb1cd1ca3"
)


@pytest.mark.skipif(
    not (TOKENIZER_PATH / "tokenizer.json").exists(),
    reason="Pinned real tokenizer has not been downloaded locally",
)
def test_real_qwen_tokenizer_multilingual_and_boundaries():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, local_files_only=True
    )
    encoder = PromptEncoder(tokenizer, max_length=256)
    assert encoder.candidates == [[16, 60], [17, 60], [18, 60], [19, 60]]
    for locale, query in [
        ("us", "mouse [1]"),
        ("es", "ratón inalámbrico"),
        ("jp", "ワイヤレスマウス"),
    ]:
        prompt = encoder.encode(
            dict(
                query=query,
                product_title=query,
                product_locale=locale,
                product_description="detail ] [ １２\n" * 1000,
            )
        )
        assert len(prompt) <= 254
    assert tokenizer.chat_template is not None
