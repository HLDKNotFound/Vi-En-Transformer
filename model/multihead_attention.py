import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting key padding mask and causal mask.
    """
    def __init__(self, model_dim, n_heads, context_length=256, dropout_rate=0.1, is_causal=False):
        super().__init__()
        assert model_dim % n_heads == 0, f"model_dim {model_dim} must be divisible by n_heads {n_heads}"

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_causal = is_causal

        self.W_query = nn.Linear(model_dim, model_dim, bias=False)
        self.W_key = nn.Linear(model_dim, model_dim, bias=False)
        self.W_value = nn.Linear(model_dim, model_dim, bias=False)
        self.W_out = nn.Linear(model_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        if is_causal:
            causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", causal_mask)
        else:
            self.causal_mask = None

    def forward(self, q, k=None, v=None, key_padding_mask=None):
        """
        q: (batch, seq_len_q, model_dim)
        k: (batch, seq_len_kv, model_dim)
        v: (batch, seq_len_kv, model_dim)
        key_padding_mask: (batch, seq_len_kv) boolean mask, True where pad
        """
        if k is None or v is None:
            k = q
            v = q

        batch_size, n_tokens_q, _ = q.shape
        n_tokens_kv = k.shape[1]

        # Project and reshape: (batch, n_heads, seq_len, head_dim)
        Q = self.W_query(q).view(batch_size, n_tokens_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(k).view(batch_size, n_tokens_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(v).view(batch_size, n_tokens_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, n_heads, n_q, n_kv)

        # Apply causal mask
        if self.is_causal and self.causal_mask is not None:
            c_mask = self.causal_mask[:n_tokens_q, :n_tokens_kv]
            scores = scores.masked_fill(c_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask shape: (batch, n_tokens_kv), expand to (batch, 1, 1, n_tokens_kv)
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        # In case all keys were masked, avoid NaN by zeroing out
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (batch, n_heads, n_q, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, n_tokens_q, self.model_dim)

        return self.W_out(context)