import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd

class MRILongitudinalDataset(Dataset):
    """
    Loads MRI triplets for longitudinal fusion training:
        (pre_scan, post_scan) -> gt_middle_scan

    Returns:
        (pre_tensor, post_tensor), gt_tensor
    """

    def __init__(self, csv_path, size=128):
        self.df = pd.read_csv(csv_path)
        self.size = size

    def __len__(self):
        return len(self.df)

    def load_and_preprocess(self, path):
        # 1. Load MRI
        img = nib.load(path).get_fdata().astype(np.float32)

        # 2. Z-score normalization
        mean = img.mean()
        std = img.std() if img.std() > 0 else 1e-8
        img = (img - mean) / std

        # 3. To tensor (C, D, H, W) = (1, D, H, W)
        img = torch.from_numpy(img).unsqueeze(0)  # add channel dim

        # 4. Resize to (1, 128, 128, 128) if needed
        if img.shape[1:] != (self.size, self.size, self.size):
            img = F.interpolate(
                img.unsqueeze(0),  # add batch
                size=(self.size, self.size, self.size),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pre_path   = row["first_path"]
        post_path  = row["last_path"]
        gt_path    = row["middle_path"]

        pre  = self.load_and_preprocess(pre_path)
        post = self.load_and_preprocess(post_path)
        gt   = self.load_and_preprocess(gt_path)

        return (pre, post), gt
