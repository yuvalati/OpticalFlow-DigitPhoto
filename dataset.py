# dataset.py

import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class FlowDataset(Dataset):
    """
    A custom Dataset that:
      1) Finds all valid (im0, im1, disp0) triplets.
      2) Determines the smallest (height, width) across the entire dataset.
      3) For each triplet, center-crops them to (target_h, target_w).
      4) Splits that cropped region into 4 quadrants (top-left, top-right, bottom-left, bottom-right).
         => Each quadrant is returned as a separate sample.
      5) **Downsamples the ground-truth** to 1/4 scale so it matches FlowNetSimple's output resolution.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.transform = transform

        # 1) Collect all *im0.png paths
        self.im0_paths = sorted(glob.glob(os.path.join(self.images_dir, '*_im0.png')))

        # Build a list of triplets: (im0_path, im1_path, disp0_path).
        self.samples = []
        for im0_path in self.im0_paths:
            base_name = os.path.basename(im0_path)
            pair_prefix = base_name.replace('_im0.png', '')
            im1_path = os.path.join(self.images_dir, f"{pair_prefix}_im1.png")
            disp_path = os.path.join(self.gt_dir, f"{pair_prefix}_disp0.jpg")

            if os.path.isfile(im1_path) and os.path.isfile(disp_path):
                self.samples.append((im0_path, im1_path, disp_path))

        # 2) Find the smallest (height, width)
        self.min_h = float('inf')
        self.min_w = float('inf')
        for (im0_path, im1_path, disp_path) in self.samples:
            with Image.open(im0_path) as im0:
                w, h = im0.size
                if h < self.min_h: self.min_h = h
                if w < self.min_w: self.min_w = w
            with Image.open(im1_path) as im1:
                w, h = im1.size
                if h < self.min_h: self.min_h = h
                if w < self.min_w: self.min_w = w
            with Image.open(disp_path) as disp0:
                w, h = disp0.size
                if h < self.min_h: self.min_h = h
                if w < self.min_w: self.min_w = w

        self.target_h = int(self.min_h)
        self.target_w = int(self.min_w)
        print(f"--> Found {len(self.samples)} valid pairs.")
        print(f"--> The smallest H,W across all images = ({self.target_h}, {self.target_w}).")

        self.total_subsamples = 4 * len(self.samples)

    def __len__(self):
        return self.total_subsamples

    def __getitem__(self, idx):
        base_idx = idx // 4
        sub_idx = idx % 4
        im0_path, im1_path, disp_path = self.samples[base_idx]

        # Load images
        im0_pil = Image.open(im0_path).convert('RGB')
        im1_pil = Image.open(im1_path).convert('RGB')
        disp0_pil = Image.open(disp_path).convert('L')  # single-channel JPG

        # Convert to torch float tensors: [C, H, W]
        im0_t = torch.from_numpy(np.array(im0_pil)).float().permute(2, 0, 1)
        im1_t = torch.from_numpy(np.array(im1_pil)).float().permute(2, 0, 1)
        disp0_t = torch.from_numpy(np.array(disp0_pil)).float().unsqueeze(0)

        # Scale images to [0..1]
        im0_t /= 255.0
        im1_t /= 255.0
        # If disp0 is in [0..255], do a scale if desired:
        # disp0_t /= 16  # example

        # 1) Center-crop
        im0_t = self._center_crop_tensor(im0_t, self.target_h, self.target_w)
        im1_t = self._center_crop_tensor(im1_t, self.target_h, self.target_w)
        disp0_t = self._center_crop_tensor(disp0_t, self.target_h, self.target_w)

        # 2) Quadrant split
        im0_sub = self._get_quadrant(im0_t, sub_idx)
        im1_sub = self._get_quadrant(im1_t, sub_idx)
        disp0_sub = self._get_quadrant(disp0_t, sub_idx)

        # 3) **Downsample the ground-truth** to 1/4 scale
        # shape is [1, subH, subW]. We'll do bilinear downsample
        subH = disp0_sub.shape[1]
        subW = disp0_sub.shape[2]
        newH = subH // 4  # integer division
        newW = subW // 4
        # We do a mini-batch dimension of 1 for interpolate
        disp0_sub_4 = F.interpolate(disp0_sub.unsqueeze(0),
                                    size=(newH, newW),
                                    mode='bilinear',
                                    align_corners=False)
        disp0_sub_4 = disp0_sub_4.squeeze(0)  # back to [1, newH, newW]

        sample = {
            'im0': im0_sub,                 # [3, subH, subW]
            'im1': im1_sub,                 # [3, subH, subW]
            'disp0': disp0_sub_4           # [1, subH/4, subW/4]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _center_crop_tensor(self, tensor_img, crop_h, crop_w):
        _, H, W = tensor_img.shape
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        return tensor_img[:, top:top+crop_h, left:left+crop_w]

    def _get_quadrant(self, tensor_img, sub_idx):
        _, H, W = tensor_img.shape
        half_h = H // 2
        half_w = W // 2
        if sub_idx == 0:  # top-left
            return tensor_img[:, 0:half_h, 0:half_w]
        elif sub_idx == 1:  # top-right
            return tensor_img[:, 0:half_h, half_w:W]
        elif sub_idx == 2:  # bottom-left
            return tensor_img[:, half_h:H, 0:half_w]
        else:               # bottom-right
            return tensor_img[:, half_h:H, half_w:W]

def demo_loader(dataset_path):
    ds = FlowDataset(root_dir=dataset_path, transform=None)
    print("Number of total sub-samples (4x each pair):", len(ds))

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    for batch_idx, sample in enumerate(loader):
        im0, im1, disp0 = sample['im0'], sample['im1'], sample['disp0']
        print(f"Batch {batch_idx}:")
        print(f"  im0 shape = {im0.shape}  (B,C,H_sub,W_sub)")
        print(f"  im1 shape = {im1.shape}")
        print(f"  disp0 shape = {disp0.shape}  (B,1,H_sub/4,W_sub/4)")
        break

if __name__ == '__main__':
    dataset_path = 'dataset'
    demo_loader(dataset_path)
