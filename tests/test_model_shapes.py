# tests/test_model_shapes.py
import torch
from src.model.gpt2_model import GPTModel_Torch

def test_forward_shape():
    vocab_size = 100
    model = GPTModel_Torch(
        vocab_size=vocab_size,
        max_len=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )

    x = torch.randint(0, vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, vocab_size)
