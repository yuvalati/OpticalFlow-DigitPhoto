import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class FlowDataset(Dataset):
    """
    A custom Dataset that loads image pairs and their corresponding disp0 ground truth.
    Assumes:
      - images/ contains *_im0.png and *_im1.png
      - GT/ contains *_disp0.jpg
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the main dataset folder, which has 'images/' and 'GT/'.
            transform (callable, optional): Optional transform to apply on the sample.
        """
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
            # Example base_name: "Backpack_imp_im0.png"
            # Remove "_im0.png" to get "Backpack_imp"
            pair_prefix = base_name.replace('_im0.png', '')

            # Construct the second image path
            im1_path = os.path.join(self.images_dir, f"{pair_prefix}_im1.png")
            # Construct the GT path for disp0 (JPG!)
            disp_path = os.path.join(self.gt_dir, f"{pair_prefix}_disp0.jpg")

            # Check that im1 & disp0 exist
            if os.path.isfile(im1_path) and os.path.isfile(disp_path):
                self.samples.append((im0_path, im1_path, disp_path))
            else:
                # You can print a warning if you like
                # print(f"Missing pair or GT for {im0_path}: {im1_path}, {disp_path}")
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im0_path, im1_path, disp_path = self.samples[idx]

        # Load images
        im0 = Image.open(im0_path).convert('RGB')
        im1 = Image.open(im1_path).convert('RGB')
        disp0 = Image.open(disp_path).convert('L')  # single-channel JPG

        # Convert to tensors:  [C,H,W] floating [0..255]
        im0 = torch.from_numpy(np.array(im0)).float().permute(2, 0, 1)
        im1 = torch.from_numpy(np.array(im1)).float().permute(2, 0, 1)
        disp0 = torch.from_numpy(np.array(disp0)).float().unsqueeze(0)  # [1,H,W]

        # Normalize images [0..1]
        im0 /= 255.0
        im1 /= 255.0

        # If disp0 is also in [0..255], you might apply a scale factor here
        # disp0 /= some_factor

        sample = {
            'im0': im0,       # [3,H,W]
            'im1': im1,       # [3,H,W]
            'disp0': disp0    # [1,H,W]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def demo_loader(dataset_path):
    ds = FlowDataset(root_dir=dataset_path, transform=None)
    print("Number of samples found:", len(ds))

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    for batch_idx, sample in enumerate(loader):
        im0, im1, disp0 = sample['im0'], sample['im1'], sample['disp0']
        print(f"Batch {batch_idx}:")
        print(f"  im0 shape = {im0.shape}  (B,C,H,W)")
        print(f"  im1 shape = {im1.shape}")
        print(f"  disp0 shape = {disp0.shape}")
        break  # just show the first batch

if __name__ == '__main__':
    dataset_path = 'dataset'
    demo_loader(dataset_path)
