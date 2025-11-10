#!/usr/bin/env python3
"""
MedVAE 3D â€” single-file trainer + encoder export

Minimal deps: torch, nibabel, numpy, tqdm
Optional: set CUDA_VISIBLE_DEVICES before running.

Examples
--------
# Train from scratch
python medvae3d.py \
  --data-glob "/path/to/preprocessed/**/*.nii.gz" \
  --outdir runs_medvae3d --size 128 --batch 2 --steps 100000 \
  --lr 2e-4 --kl-weight 1e-4 --save-every 2000

# Export encoder weights from a checkpoint
python medvae3d.py --export-encoder \
  --ckpt runs_medvae3d/medvae3d_100000.pt \
  --encoder-out runs_medvae3d/encoder_only_100k.pt

# Load encoder later (feature extractor)
enc = Encoder3D(z_dim=256); enc.load_state_dict(torch.load("runs_medvae3d/encoder_only_100k.pt"))
enc.eval()
"""
import os, glob, argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------- Model -----------------------
class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.in1   = nn.InstanceNorm3d(ch, affine=True)
        self.act1  = nn.PReLU()
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.in2   = nn.InstanceNorm3d(ch, affine=True)
    def forward(self, x):
        h = self.act1(self.in1(self.conv1(x)))
        h = self.in2(self.conv2(h))
        return torch.relu(h + x)

class Encoder3D(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        ch = [32, 64, 128, 256]
        self.stem  = nn.Sequential(nn.Conv3d(1, ch[0], 3, padding=1), nn.InstanceNorm3d(ch[0], affine=True), nn.PReLU())
        self.down1 = nn.Sequential(nn.Conv3d(ch[0], ch[1], 4, stride=2, padding=1), nn.InstanceNorm3d(ch[1], affine=True), nn.PReLU(), ResBlock3D(ch[1]))
        self.down2 = nn.Sequential(nn.Conv3d(ch[1], ch[2], 4, stride=2, padding=1), nn.InstanceNorm3d(ch[2], affine=True), nn.PReLU(), ResBlock3D(ch[2]))
        self.down3 = nn.Sequential(nn.Conv3d(ch[2], ch[3], 4, stride=2, padding=1), nn.InstanceNorm3d(ch[3], affine=True), nn.PReLU(), ResBlock3D(ch[3]))
        self.pool  = nn.AdaptiveAvgPool3d(1)
        self.mu     = nn.Linear(ch[3], z_dim)
        self.logvar = nn.Linear(ch[3], z_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        h = self.pool(x).flatten(1)
        mu, logvar = self.mu(h), self.logvar(h)
        return mu, logvar

class Decoder3D(nn.Module):
    def __init__(self, z_dim=256, base=256, size=128):
        super().__init__()
        self.size = size
        self.base = base
        d = size // 8  # expect divisible by 8
        self.fc   = nn.Linear(z_dim, base*d*d*d)
        self.up1  = nn.Sequential(nn.ConvTranspose3d(base, 128, 2, stride=2), nn.InstanceNorm3d(128, affine=True), nn.PReLU(), ResBlock3D(128))
        self.up2  = nn.Sequential(nn.ConvTranspose3d(128, 64, 2, stride=2), nn.InstanceNorm3d(64, affine=True), nn.PReLU(), ResBlock3D(64))
        self.up3  = nn.Sequential(nn.ConvTranspose3d(64, 32, 2, stride=2), nn.InstanceNorm3d(32, affine=True), nn.PReLU(), ResBlock3D(32))
        self.head = nn.Conv3d(32, 1, 1)
    def forward(self, z):
        d = self.size // 8
        h = self.fc(z).view(z.size(0), self.base, d, d, d)
        h = self.up1(h); h = self.up2(h); h = self.up3(h)
        xhat = torch.tanh(self.head(h))  # [-1,1]
        return xhat

class VAE3D(nn.Module):
    def __init__(self, z_dim=256, size=128):
        super().__init__()
        self.enc = Encoder3D(z_dim)
        self.dec = Decoder3D(z_dim, base=256, size=size)
    def forward(self, x):
        mu, logvar = self.enc(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        xhat = self.dec(z)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return xhat, kl


# ----------------------- Data -----------------------
class MRIVolDataset(Dataset):
    def __init__(self, globpat, size):
        self.paths = sorted(glob.glob(globpat, recursive=True))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No NIfTI found for pattern: {globpat}")
        self.size = size
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        vol = nib.load(p)
        arr = vol.get_fdata().astype(np.float32)
        # normalize within brain (assumes skull-stripped); fallback to global if empty
        pos = arr[arr > 0]
        m, s = (pos.mean(), pos.std() + 1e-6) if pos.size > 0 else (arr.mean(), arr.std() + 1e-6)
        arr = (arr - m) / s
        arr = np.clip(arr, -3, 3); arr = (arr + 3) / 6.0  # [0,1]
        # center pad/crop to cube (size^3)
        z, y, x = arr.shape
        t = self.size
        pad = lambda L: max(0, t - L)
        arr = np.pad(
            arr,
            ((pad(z)//2, pad(z) - pad(z)//2), (pad(y)//2, pad(y) - pad(y)//2), (pad(x)//2, pad(x) - pad(x)//2)),
            mode='constant'
        )
        z, y, x = arr.shape
        cz, cy, cx = z//2, y//2, x//2
        arr = arr[cz - t//2: cz + t//2, cy - t//2: cy + t//2, cx - t//2: cx + t//2]
        v = torch.from_numpy(arr[None, ...])  # (1, D, H, W)
        v = v * 2 - 1  # [-1,1]
        return v


# ----------------------- Train / Export -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = MRIVolDataset(args.data_glob, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=(device.type=="cuda"), drop_last=True)

    model = VAE3D(z_dim=args.z_dim, size=args.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    os.makedirs(args.outdir, exist_ok=True)
    step = 0
    pbar = tqdm(total=args.steps, ncols=100)
    model.train()
    while step < args.steps:
        for v in dl:
            v = v.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                xhat, kl = model(v)
                rec = F.mse_loss(xhat, v)
                loss = rec + args.kl_weight * kl
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                pbar.set_description(f"step {step} loss {loss.item():.4f} rec {rec.item():.4f} kl {kl.item():.4f}")
            if step % args.save_every == 0 and step > 0:
                ck = {"model": model.state_dict(), "step": step, "size": args.size, "z_dim": args.z_dim}
                torch.save(ck, os.path.join(args.outdir, f"medvae3d_{step:06d}.pt"))
            step += 1
            pbar.update(1)
            if step >= args.steps:
                break
    ck = {"model": model.state_dict(), "step": step, "size": args.size, "z_dim": args.z_dim}
    torch.save(ck, os.path.join(args.outdir, f"medvae3d_{step:06d}.pt"))

def export_encoder(args):
    if not args.ckpt:
        raise ValueError("--ckpt is required with --export-encoder")
    state = torch.load(args.ckpt, map_location="cpu")
    z_dim = state.get("z_dim", 256)
    size  = state.get("size", 128)
    model = VAE3D(z_dim=z_dim, size=size)
    model.load_state_dict(state["model"])
    enc_w = model.enc.state_dict()
    os.makedirs(os.path.dirname(args.encoder_out) or ".", exist_ok=True)
    torch.save(enc_w, args.encoder_out)
    print(f"Saved encoder weights to: {args.encoder_out}")


# ----------------------- CLI -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", type=str, help="Glob for NIfTI files (preprocessed)")
    ap.add_argument("--outdir", type=str, default="runs_medvae3d")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl-weight", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--export-encoder", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--encoder-out", type=str, default="encoder_only.pt")
    args = ap.parse_args()

    if args.export_encoder:
        export_encoder(args)
    else:
        if not args.data_glob:
            raise SystemExit("--data-glob is required for training")
        train(args)