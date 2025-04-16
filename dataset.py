import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf


def build_dataset(root_dir):
    """
        Process and create a "clean" dataset

        :param root_dir: directory of dataset folder, containing images (inputs) and ground truths maps (outputs)
        :type root_dir: str

        :return: X_final, Y_final - X for input dataset and Y for matched output (GT) dataset
    """

    images_dir = os.path.join(root_dir, 'images')
    gt_dir = os.path.join(root_dir, 'GT')

    # Collect pairs of im0 and im1, and matching ground truth disp0
    im0_paths = sorted(glob.glob(os.path.join(images_dir, '*_im0.png')))
    samples = []
    for im0_path in im0_paths:
        base_name = os.path.basename(im0_path)
        pair_prefix = base_name.replace('_im0.png', '')
        im1_path = os.path.join(images_dir, f"{pair_prefix}_im1.png")
        disp_path = os.path.join(gt_dir, f"{pair_prefix}_disp0.jpg")
        if os.path.isfile(im1_path) and os.path.isfile(disp_path):
            samples.append((im0_path, im1_path, disp_path))

    print(f"Found {len(samples)} valid pairs.")

    # Determine smallest image size (H,W)
    min_h, min_w = float('inf'), float('inf')
    for (im0_path, im1_path, disp_path) in samples:
        with Image.open(im0_path) as im0:
            w, h = im0.size
            min_h = min(min_h, h)
            min_w = min(min_w, w)
        with Image.open(im1_path) as im1:
            w, h = im1.size
            min_h = min(min_h, h)
            min_w = min(min_w, w)
        with Image.open(disp_path) as disp0:
            w, h = disp0.size
            min_h = min(min_h, h)
            min_w = min(min_w, w)

    min_h, min_w = int(min_h), int(min_w)
    print(f"Smallest H,W = ({min_h}, {min_w})")

    # Dividing each image into 4 pieces
    X_list = []
    Y_list = []

    def center_crop(np_img, crop_h, crop_w):
        H, W, C = np_img.shape
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        return np_img[top:top+crop_h, left:left+crop_w, :]

    def get_quadrant(np_img, sub_idx):
        H, W, C = np_img.shape
        half_h = H // 2
        half_w = W // 2
        # top-left
        if sub_idx == 0:
            return np_img[0:half_h, 0:half_w, :]
        # top-right
        elif sub_idx == 1:
            return np_img[0:half_h, half_w:W, :]
        # bottom-left
        elif sub_idx == 2:
            return np_img[half_h:H, 0:half_w, :]
        # bottom-right
        else:
            return np_img[half_h:H, half_w:W, :]

    for (im0_path, im1_path, disp_path) in samples:
        # Load images
        im0_pil = Image.open(im0_path).convert('RGB')
        im1_pil = Image.open(im1_path).convert('RGB')
        disp_pil = Image.open(disp_path).convert('L')  # single-channel

        # Convert images to Numpy arrays
        im0_np = np.array(im0_pil, dtype=np.float32) / 255.0  # [H,W,3]
        im1_np = np.array(im1_pil, dtype=np.float32) / 255.0  # [H,W,3]
        disp_np = np.array(disp_pil, dtype=np.float32)        # [H,W] in [0..255?]

        disp_np = disp_np[..., np.newaxis]

        # shape (min_h, min_w, 3)
        im0_np = center_crop(im0_np, min_h, min_w)
        im1_np = center_crop(im1_np, min_h, min_w)
        # shape (min_h, min_w, 1)
        disp_np = center_crop(disp_np, min_h, min_w)

        for sub_idx in range(4):
            # (H_sub, W_sub, 3)
            im0_sub = get_quadrant(im0_np, sub_idx)
            im1_sub = get_quadrant(im1_np, sub_idx)
            # (H_sub, W_sub, 1)
            disp_sub = get_quadrant(disp_np, sub_idx)

            # Downsample disp by 4 as they did in the article (H_sub/4, W_sub/4, 1)
            disp_tensor = tf.constant(disp_sub)
            disp_tensor = tf.image.resize(
                disp_tensor[tf.newaxis, ...],
                size=(disp_sub.shape[0]//4, disp_sub.shape[1]//4),
                method='bilinear',
                antialias=True
            )
            disp_ds = disp_tensor[0].numpy()  # => (H_sub/4, W_sub/4, 1)
            X_sub = np.concatenate([im0_sub, im1_sub], axis=-1)

            X_list.append(X_sub)      # inputs shape (H_sub, W_sub, 6)
            Y_list.append(disp_ds)    # outputs shape (H_sub/4, W_sub/4, 1)

    # Convert all to final Numpy arrays
    X_final = np.array(X_list, dtype=np.float32)
    Y_final = np.array(Y_list, dtype=np.float32)

    print("Final shapes:")
    print(" X (inputs):", X_final.shape, "(N, H_sub, W_sub, 6)")
    print(" Y (outputs):", Y_final.shape, "(N, H_sub/4, W_sub/4, 1)")
    return X_final, Y_final


if __name__ == "__main__":
    dataset_path = 'dataset'
    X, Y = build_dataset(dataset_path)
    print("input shape =", X.shape, "output shape =", Y.shape)
