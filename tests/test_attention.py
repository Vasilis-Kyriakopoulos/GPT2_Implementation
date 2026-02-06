import torch

from src.model.gpt2_blocks import MultiHeadAttentionCustom


def test_attention_output_shape():
    attn = MultiHeadAttentionCustom(embed_dim=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 16)

    y = attn(x)

    assert y.shape == (2, 8, 16)


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = MultiHeadAttentionCustom(embed_dim=8, num_heads=2, dropout=0.0)
    attn.eval()

    x_original = torch.randn(1, 6, 8)
    x_changed_future = x_original.clone()
    x_changed_future[:, 4:, :] = torch.randn(1, 2, 8)

    with torch.no_grad():
        y_original = attn(x_original)
        y_changed = attn(x_changed_future)
    assert torch.allclose(y_original[:, :4, :], y_changed[:, :4, :], atol=1e-6)
