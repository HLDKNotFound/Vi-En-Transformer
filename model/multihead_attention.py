import torch
import torch.nn as nn

import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, model_dim, n_heads, context_length, 
                 dropout_rate=0.1, mask=False):
        super().__init__()

        assert model_dim % n_heads == 0, f'model_dim {model_dim} must be divided by n_heads {n_heads}'

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_query = nn.Linear(model_dim, model_dim, 
                                 bias=False)
        # Query Weight
        self.W_key = nn.Linear(model_dim, model_dim, 
                               bias=False)
        # Key Weight
        self.W_value = nn.Linear(model_dim, model_dim, 
                                 bias=False)
        # Value Weight

        self.W_out = nn.Linear(model_dim, model_dim, 
                               bias=False)
        # Output Projection
        
        self.dropout = nn.Dropout(dropout_rate)
        if mask:
            causal_mask = torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            ).bool()
            self.register_buffer('causal_mask', causal_mask)
        else:
            self.causal_mask = None
        # Casual mask

    def forward(self, X_query, X_key=None, X_value=None):
        if X_key is None or X_value is None:
            X_key = X_query
            X_value = X_query

        batch_size, n_tokens_q = X_query.shape[:2]
        n_tokens_kv = X_key.shape[1]

        queries = self.W_query(X_query) 
        # (batch_size, model_dim, model_dim)
        queries = queries.view(batch_size, n_tokens_q, self.n_heads, self.head_dim) 
        # (batch_size, n_tokens, n_heads, head_dim)
        queries = queries.transpose(1, 2) 
        # (batch_size, n_heads, n_tokens, head_dim)

        keys = self.W_key(X_key) 
        # (batch_size, model_dim, model_dim)
        keys = keys.view(batch_size, n_tokens_kv, self.n_heads, self.head_dim) 
        # (batch_size, n_tokens, n_heads, head_dim)
        keys = keys.transpose(1, 2) 
        # (batch_size, n_tokens, n_heads, head_dim)

        values = self.W_value(X_value) 
        # (batch_size, model_dim, model_dim)
        values = values.view(batch_size, n_tokens_kv, self.n_heads, self.head_dim) 
        # (batch_size, n_tokens, n_heads, head_dim)
        values = values.transpose(1, 2) 
        # (batch_size, n_tokens, n_heads, head_dim)

        attn_scores = queries @ keys.transpose(2, 3)
        # (batch_size, n_heads, n_tokens, n_tokens) 
        # = (batch_size, n_heads, n_tokens, head_dim) @ (batch_size, n_heads, head_dim, n_tokens)
        if self.causal_mask is not None:
            cur_mask = self.causal_mask[:n_tokens_q, :n_tokens_kv]
            attn_scores = attn_scores.masked_fill(
                cur_mask,
                float('-inf')
            )
        attn_weights = torch.softmax(attn_scores / self.scale, dim=-1) 
        # (batch_size, n_heads, n_tokens, n_tokens)
        attn_weights = self.dropout(attn_weights) 
        # (batch_size, n_heads, n_tokens, n_tokens) 

        context_vec = (attn_weights @ values).transpose(1, 2)
        # (batch_size, n_heads, n_tokens, head_dim) 
        # = (batch_size, n_heads, n_tokens, n_tokens) @ (batch_size, n_tokens, n_tokens, head_dim)
        # => (batch_size, n_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(batch_size, n_tokens_q, self.model_dim)
        # (batch_size, n_tokens, model_dim)

        return self.W_out(context_vec) # (batch_size, n_token, model_dim)