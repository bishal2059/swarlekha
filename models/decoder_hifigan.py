# This is a compact HiFi-GAN like generator. For production, use the original HiFi-GAN repo.
import torch
import torch.nn as nn


class SimpleHiFiGAN(nn.Module):
    def __init__(self, input_dim=1024, upsample_scales=[8,8,3], out_channels=1):
        super().__init__()
        layers = []
        prev = input_dim
        for s in upsample_scales:
            layers.append(nn.ConvTranspose1d(prev, prev//2, s, stride=s))
            layers.append(nn.LeakyReLU())
            prev = prev//2
        layers.append(nn.Conv1d(prev, out_channels, 7, padding=3))
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        # x: (B, latent_dim, T)
        return self.net(x)