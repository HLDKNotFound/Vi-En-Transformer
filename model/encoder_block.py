import torch
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .moe import MoEFeedForward
from .norm import LayerNorm

class EncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Block with Mixture of Experts (MoE).
    """
    def __init__(self, model_dim, n_heads, context_length, ff_dim,
                 num_experts=5, top_k=2, dropout_rate=0.1):
        super().__init__()
        # Self-Attention
        self.norm_1 = LayerNorm(model_dim)
        self.attention = MultiHeadAttention(model_dim, n_heads, context_length,
                                            dropout_rate=dropout_rate, is_causal=False)
        self.dropout_1 = nn.Dropout(dropout_rate)

        # MoE Feed-Forward
        self.norm_2 = LayerNorm(model_dim)
        self.moe = MoEFeedForward(model_dim, ff_dim, num_experts=num_experts,
                                  top_k=top_k, dropout_rate=dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x, src_mask=None):
        # Pre-LN Self-Attention
        normed = self.norm_1(x)
        attn = self.attention(normed, normed, normed, key_padding_mask=src_mask)
        x = x + self.dropout_1(attn)

        # Pre-LN MoE
        normed = self.norm_2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + self.dropout_2(moe_out)

        return x, aux_loss