import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from .feedforward import FeedForward
from .norm import LayerNorm

class EncoderBlock(nn.Module):

    def __init__(self, model_dim, n_heads, context_length, ff_dim, 
                 dropout_rate=0.1):
        super().__init__()

        # Self-Attention
        self.attention = MultiHeadAttention(model_dim, n_heads, context_length, 
                                            dropout_rate)
        self.norm_layer_1 = LayerNorm(model_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Feed Forward
        self.feedforward = FeedForward(model_dim, ff_dim, 
                                       dropout_rate)
        self.norm_layer_2 = LayerNorm(model_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, X_encoder):
        attn = self.attention(X_encoder) # Attention
        attn = self.dropout_1(attn) # Dropout
        X_encoder = X_encoder + attn # Residual connection
        X_encoder = self.norm_layer_1(X_encoder) # Normalization

        # FeedForward
        ff = self.feedforward(X_encoder) # Feed Forward
        ff = self.dropout_2(ff) # Dropout
        X_encoder = X_encoder + ff # Residual connection
        X_encoder = self.norm_layer_2(X_encoder) # Normalization

        return X_encoder