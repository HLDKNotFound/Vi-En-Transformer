import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    Standard Feed-Forward Expert block with GELU activation.
    """
    def __init__(self, model_dim, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class MoEFeedForward(nn.Module):
    """
    Mixture of Experts (MoE) Feed-Forward layer.
    - num_experts: 5 experts
    - top_k: 2 experts selected per token (both training and inference)
    - Load balancing auxiliary loss (Switch/GShard formulation) to prevent expert collapse
    """
    def __init__(self, model_dim, ff_dim, num_experts=5, top_k=2, dropout_rate=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Router / Gating Network
        self.router = nn.Linear(model_dim, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            Expert(model_dim, ff_dim, dropout_rate) for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout_rate)

        # For monitoring expert load distribution in UI
        self.register_buffer("last_expert_counts", torch.zeros(num_experts))

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.contiguous().view(-1, self.model_dim)
        num_tokens = x_flat.size(0)

        # Compute router logits: (N_tokens, num_experts)
        router_logits = self.router(x_flat)

        # Select Top-K experts per token
        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (N_tokens, top_k)

        # Auxiliary Load Balancing Loss
        router_probs = F.softmax(router_logits, dim=-1)
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()  # (N, k, E)
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # (E,)

        if not self.training:
            self.last_expert_counts = tokens_per_expert.detach()

        # Fraction of tokens dispatched to each expert
        f_e = tokens_per_expert / (num_tokens * self.top_k + 1e-9)
        # Mean probability assigned to each expert across tokens
        P_e = router_probs.mean(dim=0)
        # Switch Transformer / GShard load balancing loss
        aux_loss = self.num_experts * torch.sum(f_e * P_e)

        # Dispatch tokens to selected experts
        out_flat = torch.zeros_like(x_flat)
        for e_idx in range(self.num_experts):
            for k in range(self.top_k):
                mask = (topk_indices[:, k] == e_idx)
                if mask.any():
                    tokens_for_expert = x_flat[mask]
                    expert_out = self.experts[e_idx](tokens_for_expert)
                    weight = topk_weights[mask, k].unsqueeze(-1)
                    out_flat[mask] += expert_out * weight

        out = self.dropout(out_flat.view(orig_shape))
        return out, aux_loss
