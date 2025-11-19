import torch
import torch.nn as nn


class ConditioningEncoder(nn.Module):
    """Attend over mel-spectrogram and produce a fixed set of embeddings (Perceiver-style).
    Minimal implementation: stacked attention blocks + linear pool to produce K embeddings.
    """
    def __init__(self, mel_dim=80, hidden=512, n_layers=6, n_heads=16, out_K=32, out_dim=1024):
        super().__init__()
        self.input_proj = nn.Linear(mel_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.resampler = nn.Linear(hidden, out_K * out_dim)
        self.out_K = out_K
        self.out_dim = out_dim


    def forward(self, mels, src_key_padding_mask=None):
        # mels: (B, T, mel_dim)
        x = self.input_proj(mels)
        x = x.permute(1,0,2) # Transformer expects (S, B, E)
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        h = h.permute(1,0,2) # (B, T, hidden)
        # pool/resample to K embeddings
        flat = self.resampler(h) # (B, T, K*out_dim)
        # mean across time, then reshape
        pooled = flat.mean(dim=1)
        out = pooled.view(h.size(0), self.out_K, self.out_dim)
        return out