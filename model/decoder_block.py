import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from .feedforward import FeedForward
from .norm import LayerNorm

class DecoderBlock(nn.Module):

    def __init__(self, model_dim, n_heads, context_length, ff_dim, 
                 dropout_rate=0.1):
        super().__init__()

        # Mask Self-Attention
        self.self_attention = MultiHeadAttention(model_dim, n_heads, context_length, 
                                                 dropout_rate, mask=True)
        self.norm_1 = LayerNorm(model_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Cross-Attention
        self.cross_attention = MultiHeadAttention(model_dim, n_heads, context_length, 
                                                  dropout_rate)
        self.norm_2 = LayerNorm(model_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Feed Forward
        self.feedforward = FeedForward(model_dim, ff_dim, 
                                       dropout_rate)
        self.norm_3 = LayerNorm(model_dim)
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, X_encoder, X_decoder):
        attn = self.self_attention(X_decoder) # Self-Attention
        attn = self.dropout_1(attn) # Dropout
        X_decoder = X_decoder + attn # Residual connection
        X_decoder = self.norm_1(X_decoder) # Norm Layer

        attn = self.cross_attention(X_decoder, X_encoder, X_encoder) # Cross-Attention
        attn = self.dropout_2(attn) # Dropout
        X_decoder = X_decoder + attn # Residual connection
        X_decoder = self.norm_2(X_decoder) # Norm Layer

        ff = self.feedforward(X_decoder) # Feed Forward
        ff = self.dropout_3(ff) # Dropout
        X_decoder = X_decoder + ff # Residual connection
        X_decoder = self.norm_3(X_decoder) # Norm Layer

        return X_decoder