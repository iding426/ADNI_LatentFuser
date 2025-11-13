import torch
import torch.nn as nn
import numpy

# Transformer-based feature fusion
# Combine two latent vectors into one latent
class FusionTransformer(nn.Module):
    def __init__(self,
                 latent_dim=256,
                 nhead=4,
                 num_layers=2,):
        super(FusionTransformer, self).__init__()
        self.latent_dim = latent_dim

        # Positional Embeddings
        self.pos = nn.Parameter(torch.randn(1, 2, latent_dim))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=latent_dim*4,
                                                   batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        
        # Final linear layer to project back to latent_dim
        self.lin = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x_1, x_2):
        x = torch.stack((x_1, x_2), dim=1)  # (B, 2, latent_dim)
        x = x + self.pos  # Add positional embeddings

        y = self.transformer_encoder(x)  # (B, 2, latent_dim)

        y = y.reshape(y.shape[0], -1)  # (B, latent_dim*2)
        y = self.lin(y)  # (B, latent_dim)
        return y  # Fused Latent - (B, latent_dim)

