import torch
import torch.nn as nn


class SimpleSpeakerEncoder(nn.Module):
    """Small speaker embedding model (educational). For strong results use H/ASP or a precomputed model.
    This model expects waveform or mel input and outputs a fixed 512-d embedding.
    """
    def __init__(self, mel_dim=80, hidden=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv1d(mel_dim, hidden, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(hidden, out_dim)
        )


    def forward(self, mels):
        # mels: (B, T, mel_dim) -> conv expects (B, C, T)
        x = mels.permute(0,2,1)
        return self.net(x)