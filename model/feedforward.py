import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim,
                dropout_rate=0.1):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim), # Upscale model_dim to ff_dim
            nn.GELU(), # GELU
            nn.Dropout(dropout_rate), # Dropout
            nn.Linear(ff_dim, model_dim), # Downscale ff_dim to model_dim
        )

    def forward(self, X):
        return self.feed_forward(X)