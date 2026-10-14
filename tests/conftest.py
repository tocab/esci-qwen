import pytest
import torch

from src.common import seed_all
from tests.helpers import tiny_model, tiny_tokenizer


@pytest.fixture
def tokenizer():
    return tiny_tokenizer()


@pytest.fixture
def model(tokenizer):
    seed_all(42)
    torch.set_num_threads(2)
    return tiny_model(len(tokenizer)).eval()
