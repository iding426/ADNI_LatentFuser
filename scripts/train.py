import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Make project root importable so this script can be run directly (python scripts/train.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import your VAE and dataset (absolute imports; project root is on sys.path)
from models.medvae3d import VAE3D, MRIVolDataset
from models.mlp_fusion import FusionMLP
from models.transformer_fusion import FusionTransformer
from scripts.adni_ds import MRILongitudinalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion-model", type=str, default="mlp", choices=["mlp", "transformer"], help="Which fusion model to train")
parser.add_argument("--vae-ckpt", type=str, required=False, help="Path to pre-trained VAE checkpoint")  
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save fusion model checkpoints")
args = parser.parse_args()

# Create checkpoint directory
checkpoint_dir = Path(args.checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True, parents=True)
print(f"Checkpoints will be saved to: {checkpoint_dir}")

# Hyperparameters
latent_dim = 256
epochs = 20
lr = 1e-4
batch_size = 2
size = 128

# Load pre-trained VAE
print(f"Loading pre-trained VAE from {args.vae_ckpt}...")
vae3d = VAE3D(z_dim=latent_dim, size=size).to(device)

if args.vae_ckpt:
    # Load checkpoint - handle both direct state_dict and wrapped checkpoint formats
    vae_checkpoint = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    if isinstance(vae_checkpoint, dict) and 'model' in vae_checkpoint:
        # Checkpoint is wrapped (contains 'model', 'step', etc.)
        vae3d.load_state_dict(vae_checkpoint['model'])
        print(f"VAE loaded from checkpoint at step {vae_checkpoint.get('step', 'unknown')}")
    else:
        # Direct state_dict
        vae3d.load_state_dict(vae_checkpoint)
        print("VAE loaded from checkpoint")
else:
    print("Using pretrained MedVAE3D weights (no checkpoint provided)")

vae3d.eval()  # Set to evaluation mode

# Freeze VAE parameters
for param in vae3d.parameters():
    param.requires_grad = False

if args.fusion_model == "mlp":
    fusion_model = FusionMLP(latent_dim=256).to(device)
elif args.fusion_model == "transformer":
    fusion_model = FusionTransformer(latent_dim=256).to(device)
else:
    raise ValueError(f"Unknown fusion model: {args.fusion_model}")

fusion_model.train()

optimizer = torch.optim.Adam(fusion_model.parameters(), lr)
loss_fn = nn.MSELoss()

# Dataset and DataLoader
train_csv = "./data/triplets_train.csv"
dataset = MRILongitudinalDataset(train_csv, size=size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(epochs):
    running_loss = 0.0
    
    for (pre, post), gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        pre, post, gt = pre.to(device), post.to(device), gt.to(device)

        # Encode
        mu_pre, logvar_pre = vae3d.enc(pre)
        mu_post, logvar_post = vae3d.enc(post)

        mu_gt, logvar_gt = vae3d.enc(gt)

        # Fusion with Transformer
        fused_latent = fusion_model(mu_pre, mu_post)

        # Loss between fused and ground truth latent
        loss = loss_fn(fused_latent, mu_gt)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pre.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    # Save checkpoint for this epoch
    checkpoint_path = checkpoint_dir / f"{args.fusion_model}_fusion_epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

# Save final model
final_model_path = checkpoint_dir / f"{args.fusion_model}_fusion_final.pt"
torch.save(fusion_model.state_dict(), final_model_path)