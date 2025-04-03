import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MovingObjectDisparityDataset(Dataset):
    """
    Custom Dataset for loading pairs of images and a chosen disparity map.
    The folder structure:
        dataset/
          ├─ images/
          │   ├─ object_state_im0.png (or .jpg, etc.)
          │   ├─ object_state_im1.png
          └─ GT/
              ├─ object_state_disp0.png (or .jpg, etc.)
              ├─ object_state_disp0y.png
              ...
    Example filenames:
      - Backpack_perfect_im0.png
      - Backpack_perfect_im1.png
      - Backpack_per_disp0.jpg
      - Backpack_per_disp0y.jpg
      - etc.
    """

    def __init__(self,
                 root_dir,
                 transform=None,
                 use_disp_name='disp0'):
        """
        Args:
            root_dir (str): Path to the dataset folder (containing 'images' and 'GT').
            transform (callable, optional): Optional transform to be applied on a sample.
            use_disp_name (str): Which disparity map pattern to use (e.g. 'disp0', 'disp0y', 'disp0-cd').
        """
        super().__init__()
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.transform = transform
        self.use_disp_name = use_disp_name  # which GT map we want

        # Gather all im0 files
        # For instance, anything ending in '_im0.*'
        self.im0_list = sorted(glob.glob(os.path.join(self.images_dir, '*_im0.*')))

        # We will create a list of tuples (im0_path, im1_path, disp_path)
        self.samples = []
        for im0_path in self.im0_list:
            # Example:  Backpack_perfect_im0.png
            # we want to also find Backpack_perfect_im1.png
            file_name = os.path.basename(im0_path)
            # remove '_im0' from name
            common_prefix = file_name.rsplit('_im0', 1)[0]  # e.g. "Backpack_perfect"
            ext = file_name.rsplit('.', 1)[1]               # e.g. "png"

            # Build the corresponding im1 filename
            im1_name = f"{common_prefix}_im1.{ext}"
            im1_path = os.path.join(self.images_dir, im1_name)
            if not os.path.exists(im1_path):
                continue  # skip if we don't have a matching im1

            # Build the disparity filename in GT folder
            # For example, "Backpack_perfect_disp0.(extension?)"
            # The extension in GT might differ, so we can attempt a small search:
            disp_pattern = os.path.join(self.gt_dir, f"{common_prefix}_{self.use_disp_name}*")
            found_disp = glob.glob(disp_pattern)
            if not found_disp:
                continue  # skip if no matching GT found
            # assume the first match
            disp_path = found_disp[0]

            self.samples.append((im0_path, im1_path, disp_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im0_path, im1_path, disp_path = self.samples[idx]

        im0 = Image.open(im0_path).convert('RGB')
        im1 = Image.open(im1_path).convert('RGB')
        disp = Image.open(disp_path)

        # Convert to tensors
        # Depending on your task, you might want to keep disparity as float
        # and images scaled between [0,1].
        to_tensor = T.ToTensor()

        im0_t = to_tensor(im0)
        im1_t = to_tensor(im1)
        disp_t = to_tensor(disp)  # e.g. grayscale => shape [1, H, W]

        # If you have some custom data augmentation, you can apply it here:
        if self.transform is not None:
            im0_t, im1_t, disp_t = self.transform(im0_t, im1_t, disp_t)

        sample = {
            'im0': im0_t,        # shape [3, H, W]
            'im1': im1_t,        # shape [3, H, W]
            'disp': disp_t       # shape [1, H, W] or [H, W] if not converted to 3D
        }

        return sample
