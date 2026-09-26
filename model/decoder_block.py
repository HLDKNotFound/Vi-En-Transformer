import torch
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .moe import MoEFeedForward
from .norm import LayerNorm

class DecoderBlock(nn.Module):
    """
    Pre-LN Transformer Decoder Block with Mixture of Experts (MoE).
    """
    def __init__(self, model_dim, n_heads, context_length, ff_dim,
                 num_experts=5, top_k=2, dropout_rate=0.1):
        super().__init__()
        # Causal Self-Attention
        self.norm_1 = LayerNorm(model_dim)
        self.self_attention = MultiHeadAttention(model_dim, n_heads, context_length,
                                                 dropout_rate=dropout_rate, is_causal=True)
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Cross-Attention to Encoder output
        self.norm_2 = LayerNorm(model_dim)
        self.cross_attention = MultiHeadAttention(model_dim, n_heads, context_length,
                                                  dropout_rate=dropout_rate, is_causal=False)
        self.dropout_2 = nn.Dropout(dropout_rate)

        # MoE Feed-Forward
        self.norm_3 = LayerNorm(model_dim)
        self.moe = MoEFeedForward(model_dim, ff_dim, num_experts=num_experts,
                                  top_k=top_k, dropout_rate=dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, x_decoder, x_encoder, tar_mask=None, src_mask=None):
        # 1. Pre-LN Causal Self-Attention
        normed_dec = self.norm_1(x_decoder)
        self_attn = self.self_attention(normed_dec, normed_dec, normed_dec, key_padding_mask=tar_mask)
        x_decoder = x_decoder + self.dropout_1(self_attn)

        # 2. Pre-LN Cross-Attention
        normed_cross = self.norm_2(x_decoder)
        cross_attn = self.cross_attention(normed_cross, x_encoder, x_encoder, key_padding_mask=src_mask)
        x_decoder = x_decoder + self.dropout_2(cross_attn)

        # 3. Pre-LN MoE Feed-Forward
        normed_moe = self.norm_3(x_decoder)
        moe_out, aux_loss = self.moe(normed_moe)
        x_decoder = x_decoder + self.dropout_3(moe_out)

        return x_decoder, aux_loss