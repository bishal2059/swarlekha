import torch
import torch.nn as nn


class SimpleVQVAE(nn.Module):
    """A compact VQ-VAE for mel -> discrete codes.
    This is a compact educational implementation (not production).
    """
    def __init__(self, in_dim=80, hidden=512, codebook_size=1024, code_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv1d(in_dim, hidden, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden, code_dim, 3, padding=1)
        )
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        # decoder
        self.decoder = nn.Sequential(
        nn.Conv1d(code_dim, hidden, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden, in_dim, 3, padding=1)
        )


    def forward(self, mels):
        # mels: (B, T, C) -> transpose for conv1d
        x = mels.transpose(1,2)
        z = self.encoder(x) # (B, code_dim, T)
        z_t = z.permute(0,2,1) # (B, T, code_dim)
        # compute distance to codebook
        flat = z_t.reshape(-1, self.code_dim)
        d = (flat**2).sum(1, keepdim=True) - 2*flat@self.codebook.weight.t() + (self.codebook.weight.t()**2).sum(0,keepdim=True)
        idx = d.argmin(1)
        idx = idx.view(z_t.shape[0], z_t.shape[1])
        quantized = self.codebook(idx)
        q = quantized.permute(0,2,1)
        recon = self.decoder(q)
        recon = recon.transpose(1,2)
        return recon, idx