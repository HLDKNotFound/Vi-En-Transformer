import torch
import torch.nn as nn

from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock

class TransformerModel(nn.Module):

    def __init__(self, vocab_size, model_dim, n_heads, context_length, ff_dim, 
                 n_encoders, n_decoders, 
                 dropout_rate=0.1, device='cpu'):
        super().__init__()
        
        self.device = device

        # token embedding
        self.tokens_emb = nn.Embedding(vocab_size, model_dim)
        # position embedding
        self.position_emb = nn.Embedding(context_length, model_dim)

        # Encoder Blocks
        self.encoder = nn.ModuleList([
            EncoderBlock(model_dim, n_heads, context_length, ff_dim, 
                         dropout_rate)
            for _ in range(n_encoders)
        ])
        self.dropout_encoder = nn.Dropout(dropout_rate)

        # Decoder Blocks
        self.decoder = nn.ModuleList([
            DecoderBlock(model_dim, n_heads, context_length, ff_dim, 
                         dropout_rate)
            for _ in range(n_decoders)
        ])
        self.dropout_decoder = nn.Dropout(dropout_rate)

        self.out_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, src_idx, tar_idx):
        # Embedding source index
        encoder_tokens = self.tokens_emb(src_idx)
        encoder_indices = torch.arange(src_idx.shape[1]).to(self.device)
        encoder_pos = self.position_emb(encoder_indices)
        encoder_pos = encoder_pos.unsqueeze(0)
        X_encoder = encoder_tokens + encoder_pos
    
        # Dropout
        X_encoder = self.dropout_encoder(X_encoder)

        # Encoder Phase
        for encoder_layer in self.encoder:
            X_encoder = encoder_layer(X_encoder)

        # Embedding target index
        decoder_tokens = self.tokens_emb(tar_idx)
        decoder_indices = torch.arange(tar_idx.shape[1]).to(self.device)
        decoder_pos = self.position_emb(decoder_indices)
        decoder_pos = decoder_pos.unsqueeze(0)
        X_decoder = decoder_tokens + decoder_pos

        # Dropout
        X_decoder = self.dropout_decoder(X_decoder)

        # Decoder Phase
        for decoder_layer in self.decoder: 
            X_decoder = decoder_layer(X_encoder, X_decoder)

        logit = self.out_layer(X_decoder)

        return logit