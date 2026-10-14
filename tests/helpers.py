"""Small offline fixtures for training and data tests."""

import polars as pl
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from src.common import LABELS
from src.data import FIELDS
from src.jev import PromptEncoder
from src.models import native_qwen_kernels


def synthetic_data():
    examples, products = [], []
    words = ("apple", "orange", "grape", "melon")
    for group in range(9):
        for label, word in zip(LABELS, words):
            product_id = f"{group}-{label}"
            locale = ("us", "es", "jp")[group % 3]
            examples.append(
                dict(
                    example_id=product_id,
                    query_id=group,
                    query="fruit " + str(group),
                    product_id=product_id,
                    product_locale=locale,
                    esci_label=label,
                    large_version=1,
                    small_version=1,
                    split="test" if group == 8 else "train",
                )
            )
            products.append(
                dict(
                    product_id=product_id,
                    product_locale=locale,
                    product_title=word,
                    product_brand=None,
                    product_color="",
                    product_bullet_point="",
                    product_description="",
                )
            )
    return pl.DataFrame(examples), pl.DataFrame(products)


def tiny_tokenizer():
    # WordLevel + punctuation separation gives multi-token digit/closing-bracket suffixes.
    pre = pre_tokenizers.Whitespace()
    encoder = PromptEncoder.__new__(PromptEncoder)
    encoder.order = LABELS
    corpus = encoder.render(
        {
            field: "apple orange grape melon fruit us es jp "
            "wireless gaming mouse black ExampleBrand"
            for field in FIELDS
        }
    )
    corpus += " 1 2 3 4 5 6 7 8 9 0 Selection: [1] [2] [3] [4]"
    words = sorted({word for word, _ in pre.pre_tokenize_str(corpus)})
    vocab = {"[PAD]": 0, "[UNK]": 1, **{word: i + 2 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")


def tiny_model(vocab_size):
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=1024,
            pad_token_id=0,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
        ),
        image_token_id=vocab_size + 1,
        video_token_id=vocab_size + 2,
        vision_start_token_id=vocab_size + 3,
        vision_end_token_id=vocab_size + 4,
    )
    with native_qwen_kernels():
        return Qwen3_5ForConditionalGeneration(config)
