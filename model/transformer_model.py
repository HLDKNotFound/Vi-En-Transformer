import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock
from .norm import LayerNorm

class TransformerModel(nn.Module):
    """
    150M Parameter Sparse Mixture-of-Experts (MoE) Encoder-Decoder Transformer:
    - 6 Encoder Layers + 6 Decoder Layers
    - 5 Experts per MoE layer (Top-2 selected per token)
    - Model Dimension: 512, FF Dimension per expert: 1860, Heads: 8
    - Weight-tied token embeddings and output projection layer
    - Gradient Checkpointing support for ultra-low VRAM training (<2GB on RTX 4050)
    - Total parameters: ~150M (active ~63M per token)
    """
    def __init__(self, vocab_size, model_dim=512, n_heads=8, context_length=128,
                 ff_dim=1860, n_encoders=6, n_decoders=6, num_experts=5, top_k=2,
                 dropout_rate=0.1, pad_id=0, tie_embeddings=True, gradient_checkpointing=True, device='cpu'):
        super().__init__()
        self.device = device
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.context_length = context_length
        self.gradient_checkpointing = gradient_checkpointing

        # Embeddings
        self.tokens_emb = nn.Embedding(vocab_size, model_dim)
        self.position_emb = nn.Embedding(context_length, model_dim)
        self.dropout_emb = nn.Dropout(dropout_rate)

        # Encoder Layers
        self.encoders = nn.ModuleList([
            EncoderBlock(model_dim, n_heads, context_length, ff_dim,
                         num_experts=num_experts, top_k=top_k, dropout_rate=dropout_rate)
            for _ in range(n_encoders)
        ])
        self.encoder_norm = LayerNorm(model_dim)

        # Decoder Layers
        self.decoders = nn.ModuleList([
            DecoderBlock(model_dim, n_heads, context_length, ff_dim,
                         num_experts=num_experts, top_k=top_k, dropout_rate=dropout_rate)
            for _ in range(n_decoders)
        ])
        self.decoder_norm = LayerNorm(model_dim)

        # Output Projection Layer
        self.out_layer = nn.Linear(model_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.out_layer.weight = self.tokens_emb.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def encode(self, src_idx):
        batch_size, src_len = src_idx.shape
        src_mask = (src_idx == self.pad_id)

        positions = torch.arange(src_len, device=src_idx.device).unsqueeze(0)
        x_enc = self.dropout_emb(self.tokens_emb(src_idx) + self.position_emb(positions))

        aux_loss = 0.0
        for layer in self.encoders:
            if self.training and self.gradient_checkpointing:
                x_enc, layer_aux = checkpoint(layer, x_enc, src_mask, use_reentrant=False)
            else:
                x_enc, layer_aux = layer(x_enc, src_mask=src_mask)
            aux_loss = aux_loss + layer_aux

        x_enc = self.encoder_norm(x_enc)
        return x_enc, src_mask, aux_loss

    def forward(self, src_idx, tar_idx):
        """
        Full forward pass for training with teacher forcing and gradient checkpointing.
        """
        # 1. Encode source
        x_enc, src_mask, enc_aux_loss = self.encode(src_idx)

        # 2. Decode target
        batch_size, tar_len = tar_idx.shape
        tar_mask = (tar_idx == self.pad_id)

        positions = torch.arange(tar_len, device=tar_idx.device).unsqueeze(0)
        x_dec = self.dropout_emb(self.tokens_emb(tar_idx) + self.position_emb(positions))

        dec_aux_loss = 0.0
        for layer in self.decoders:
            if self.training and self.gradient_checkpointing:
                x_dec, layer_aux = checkpoint(layer, x_dec, x_enc, tar_mask, src_mask, use_reentrant=False)
            else:
                x_dec, layer_aux = layer(x_dec, x_enc, tar_mask=tar_mask, src_mask=src_mask)
            dec_aux_loss = dec_aux_loss + layer_aux

        x_dec = self.decoder_norm(x_dec)
        logits = self.out_layer(x_dec)

        total_aux_loss = enc_aux_loss + dec_aux_loss
        return logits, total_aux_loss

    def decode_step(self, x_enc, src_mask, cur_tar_idx):
        """
        Efficient step-by-step autoregressive decode for inference.
        """
        tar_len = cur_tar_idx.shape[1]
        tar_mask = (cur_tar_idx == self.pad_id)

        positions = torch.arange(tar_len, device=cur_tar_idx.device).unsqueeze(0)
        x_dec = self.dropout_emb(self.tokens_emb(cur_tar_idx) + self.position_emb(positions))

        for layer in self.decoders:
            x_dec, _ = layer(x_dec, x_enc, tar_mask=tar_mask, src_mask=src_mask)

        x_dec = self.decoder_norm(x_dec)
        next_token_logits = self.out_layer(x_dec[:, -1, :])
        return next_token_logits