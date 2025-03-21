import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class MiddleburyFlowDataset(Dataset):
    """
    Reads all PNG files in a single folder where files are named like:
      {Scene}_{Version}_im0.png
      {Scene}_{Version}_im1.png

    For each matching 'im0'/'im1' pair, generates 4 sub-patches:
      top-left, top-right, bottom-left, bottom-right
    so that the dataset length = (#pairs * 4).
    """

    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform  # To Tensor

        # Gather all .png files
        all_pngs = sorted(glob.glob(os.path.join(data_dir, '*.png')))

        # Build a dict with key = (scene_version), storing paths for "im0" and "im1"
        pairs_dict = {}
        for path in all_pngs:
            filename = os.path.basename(path)
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split('_')
            scene_version = '_'.join(parts[:-1])
            im_label = parts[-1]

            if scene_version not in pairs_dict:
                pairs_dict[scene_version] = {}
            pairs_dict[scene_version][im_label] = path

        self.pairs = []
        for scene_version, paths in pairs_dict.items():
            if 'im0' in paths and 'im1' in paths:
                self.pairs.append((paths['im0'], paths['im1']))

        # Each pair produces 4 sub-patches => total samples = len(self.pairs) * 4
        self.num_subpatches = 4
        self.total_samples = len(self.pairs) * self.num_subpatches

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        """
        Map the index to (pair_index, subpatch_index).
        pair_index = index // 4
        subpatch_id = index % 4
        """
        pair_index = index // self.num_subpatches
        subpatch_id = index % self.num_subpatches

        im0_path, im1_path = self.pairs[pair_index]
        img0 = Image.open(im0_path).convert('RGB')
        img1 = Image.open(im1_path).convert('RGB')

        # Both images should have the same size
        w, h = img0.size

        # Midpoints for splitting
        mid_w = w // 2
        mid_h = h // 2

        # subpatch_id: 0 => top-left, 1 => top-right, 2 => bottom-left, 3 => bottom-right
        if subpatch_id == 0:
            box = (0, 0, mid_w, mid_h)
        elif subpatch_id == 1:
            box = (mid_w, 0, w, mid_h)
        elif subpatch_id == 2:
            box = (0, mid_h, mid_w, h)
        else:  # subpatch_id == 3
            box = (mid_w, mid_h, w, h)

        # Crop both images
        img0_crop = img0.crop(box)
        img1_crop = img1.crop(box)

        sample = {
            'img0': img0_crop,
            'img1': img1_crop
        }

        # Apply transform to get Tensors (and possibly augmentations)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# transform that simply converts the two cropped PIL images to tensors
class FlowPatchTransform(object):
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        img0_pil = sample['img0']
        img1_pil = sample['img1']

        img0_tensor = self.to_tensor(img0_pil)
        img1_tensor = self.to_tensor(img1_pil)

        return {
            'img0': img0_tensor,
            'img1': img1_tensor
        }


if __name__ == "__main__":
    data_dir = "dataset"

    # Create the dataset with a transform that converts PIL images to Tensors
    dataset = MiddleburyFlowDataset(data_dir=data_dir, transform=FlowPatchTransform())

    print("Total samples:", len(dataset))  # #pairs * 4
    sample0 = dataset[0]
    print("Shape of img0:", sample0['img0'].shape)
    print("Shape of img1:", sample0['img1'].shape)

    # If you want to feed this into a model:
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for batch in loader:
    #     img0_batch = batch['img0']  # (B, 3, H/2, W/2)
    #     img1_batch = batch['img1']  # (B, 3, H/2, W/2)
    #     # pass to FlowNet, etc.
