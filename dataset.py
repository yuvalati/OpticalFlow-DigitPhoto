import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class FlowDataset(Dataset):
    """
    A custom Dataset that loads image pairs (im0, im1) and their corresponding disp0 ground truth,
    then crops them to match the smallest H,W across the entire dataset.
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

            if os.path.isfile(im1_path) and os.path.isfile(disp_path):
                self.samples.append((im0_path, im1_path, disp_path))

        # 2) Figure out the smallest (height, width) among all images in the dataset
        self.min_h = float('inf')
        self.min_w = float('inf')

        for (im0_path, im1_path, disp_path) in self.samples:
            # Check each file's size
            with Image.open(im0_path) as im0:
                w, h = im0.size  # Pillow returns (width, height)
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

        # For convenience, store the final target size as integers
        self.target_h = int(self.min_h)
        self.target_w = int(self.min_w)

        print(f"--> Found {len(self.samples)} valid pairs.")
        print(f"--> The smallest H,W across all images = ({self.target_h}, {self.target_w}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im0_path, im1_path, disp_path = self.samples[idx]

        # Load images with Pillow
        im0_pil = Image.open(im0_path).convert('RGB')
        im1_pil = Image.open(im1_path).convert('RGB')
        disp0_pil = Image.open(disp_path).convert('L')  # single-channel JPG

        # Convert to numpy => then torch
        im0_t = torch.from_numpy(np.array(im0_pil)).float().permute(2, 0, 1)  # [3,H,W]
        im1_t = torch.from_numpy(np.array(im1_pil)).float().permute(2, 0, 1)
        disp0_t = torch.from_numpy(np.array(disp0_pil)).float().unsqueeze(0)   # [1,H,W]

        # Normalize images [0..1]
        im0_t /= 255.0
        im1_t /= 255.0
        # If disp0 is in [0..255], scale if needed, e.g. disp0_t /= 16

        # 3) Center-crop to (target_h, target_w)
        im0_t = self._center_crop_tensor(im0_t, self.target_h, self.target_w)
        im1_t = self._center_crop_tensor(im1_t, self.target_h, self.target_w)
        disp0_t = self._center_crop_tensor(disp0_t, self.target_h, self.target_w)

        sample = {
            'im0': im0_t,       # [3, target_h, target_w]
            'im1': im1_t,       # [3, target_h, target_w]
            'disp0': disp0_t    # [1, target_h, target_w]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _center_crop_tensor(self, tensor_img, crop_h, crop_w):
        """
        Given a torch tensor of shape [C,H,W], center-crop it to (crop_h, crop_w).
        Half the difference is cut from each side in H and W dimensions.
        """
        _, H, W = tensor_img.shape
        # Compute top/left for center
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        return tensor_img[:, top:top+crop_h, left:left+crop_w]

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
