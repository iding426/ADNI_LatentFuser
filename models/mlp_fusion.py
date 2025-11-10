import torch
import torch.nn as nn
import numpy as np

# MLP for feature fusion
# Combine two latent vectors into one laten
class FusionMLP(nn.Module):
    def __init__(self, latent_dim = 256) -> None:
        super(FusionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x_1, x_2) -> np.ndarray:
        # Concat the two latents (can be done in batches)
        x = torch.cat((x_1, x_2), dim=-1) # (B, latent_dim*2)

        return self.net(x) # Fused Latent - (B, latent_dim)
    

