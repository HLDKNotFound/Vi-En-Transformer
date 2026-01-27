import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, model_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim) # Normalization

    def forward(self, X):
        return self.norm(X)