import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

# Make project root importable so this script can be run directly (python scripts/eval.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import your VAE and dataset (absolute imports; project root is on sys.path)
from models.medvae3d import VAE3D
from models.mlp_fusion import FusionMLP
from models.transformer_fusion import FusionTransformer
from scripts.adni_ds import MRILongitudinalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argparse
parser = argparse.ArgumentParser(description="Evaluate fusion model on test set")
parser.add_argument("--fusion-model", type=str, default="mlp", choices=["mlp", "transformer"], help="Which fusion model to evaluate")
parser.add_argument("--vae-ckpt", type=str, required=False, help="Path to pre-trained VAE checkpoint")  
parser.add_argument("--fusion-ckpt", type=str, required=True, help="Path to fusion model checkpoint")
parser.add_argument("--test-csv", type=str, required=True, help="Path to test triplets CSV")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save evaluation results")

args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Hyperparameters
latent_dim = 256
size = 128

# Load VAE model
print("Loading VAE model...")
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

vae3d.eval()

# Load fusion model
print(f"Loading {args.fusion_model} fusion model...")
if args.fusion_model == "mlp":
    fusion_model = FusionMLP(latent_dim=256).to(device)
elif args.fusion_model == "transformer":
    fusion_model = FusionTransformer(latent_dim=256).to(device)
else:
    raise ValueError(f"Unknown fusion model: {args.fusion_model}")

# Load fusion checkpoint - handle both direct state_dict and wrapped checkpoint formats
fusion_checkpoint = torch.load(args.fusion_ckpt, map_location=device, weights_only=False)
if isinstance(fusion_checkpoint, dict) and 'model_state_dict' in fusion_checkpoint:
    # Checkpoint is wrapped (from training script)
    fusion_model.load_state_dict(fusion_checkpoint['model_state_dict'])
    print(f"✓ Fusion model loaded from epoch {fusion_checkpoint.get('epoch', 'unknown')}")
else:
    # Direct state_dict
    fusion_model.load_state_dict(fusion_checkpoint)
    print("✓ Fusion model loaded")

fusion_model.eval()


# Evaluation metrics functions
def compute_mse(pred, target):
    """Compute Mean Squared Error"""
    return F.mse_loss(pred, target, reduction='mean').item()

def compute_mae(pred, target):
    """Compute Mean Absolute Error"""
    return F.l1_loss(pred, target, reduction='mean').item()

def compute_psnr(pred, target, max_val=2.0):
    """Compute Peak Signal-to-Noise Ratio (assumes data in [-1, 1] range)"""
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()

def compute_ssim_3d(pred, target, window_size=11):
    """Simplified 3D SSIM computation"""
    # Note: For production, consider using a proper SSIM library
    mu_pred = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return ssim_map.mean().item()

def compute_cosine_similarity(pred_latent, target_latent):
    """Compute cosine similarity between latent vectors"""
    return F.cosine_similarity(pred_latent, target_latent, dim=1).mean().item()


# Dataset and DataLoader
print(f"Loading test dataset from {args.test_csv}...")
test_dataset = MRILongitudinalDataset(args.test_csv, size=size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

print(f"Test dataset size: {len(test_dataset)} samples")
print(f"Starting evaluation on {device}...")

# Evaluation metrics storage
metrics = {
    'latent_mse': [],
    'latent_mae': [],
    'latent_cosine_sim': [],
    'recon_mse': [],
    'recon_mae': [],
    'recon_psnr': [],
    'recon_ssim': []
}

# Evaluation loop
with torch.no_grad():
    for batch_idx, ((pre, post), gt) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        pre, post, gt = pre.to(device), post.to(device), gt.to(device)
        
        # Encode pre and post scans
        mu_pre, logvar_pre = vae3d.enc(pre)
        mu_post, logvar_post = vae3d.enc(post)
        
        # Encode ground truth middle scan
        mu_gt, logvar_gt = vae3d.enc(gt)
        
        # Fuse latent vectors
        fused_latent = fusion_model(mu_pre, mu_post)
        
        # Latent space metrics
        latent_mse = compute_mse(fused_latent, mu_gt)
        latent_mae = compute_mae(fused_latent, mu_gt)
        latent_cosine = compute_cosine_similarity(fused_latent, mu_gt)
        
        metrics['latent_mse'].append(latent_mse)
        metrics['latent_mae'].append(latent_mae)
        metrics['latent_cosine_sim'].append(latent_cosine)
        
        # Reconstruct from fused latent
        recon_fused = vae3d.dec(fused_latent)
        
        # Image space metrics
        recon_mse = compute_mse(recon_fused, gt)
        recon_mae = compute_mae(recon_fused, gt)
        recon_psnr = compute_psnr(recon_fused, gt)
        recon_ssim = compute_ssim_3d(recon_fused, gt)
        
        metrics['recon_mse'].append(recon_mse)
        metrics['recon_mae'].append(recon_mae)
        metrics['recon_psnr'].append(recon_psnr)
        metrics['recon_ssim'].append(recon_ssim)

# Compute summary statistics
print("EVALUATION RESULTS")

results = {}
for metric_name, values in metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    results[metric_name] = {
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }
    print(f"\n{metric_name.upper()}:")
    print(f"  Mean: {mean_val:.6f} ± {std_val:.6f}")
    print(f"  Range: [{np.min(values):.6f}, {np.max(values):.6f}]")

# Save results to JSON
results_file = output_dir / f"eval_results_{args.fusion_model}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {results_file}")

# Save per-sample metrics to CSV
import pandas as pd
metrics_df = pd.DataFrame(metrics)
csv_file = output_dir / f"eval_metrics_{args.fusion_model}.csv"
metrics_df.to_csv(csv_file, index=False)
print(f"Metrics saved to: {csv_file}")


