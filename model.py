import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image


# ----------------------------------------------------------------------------------
# 1. Example Dataset Class
# ----------------------------------------------------------------------------------

class OpticalFlowDataset(torch.utils.data.Dataset):
    """
    Example dataset class that returns two images (I1, I2) and optionally flow ground truth.
    Adjust file paths and loading details to match your dataset structure.
    """

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Example: looking for image_0/*.png, image_1/*.png, and flow/*.flo
        self.img1_paths = sorted(glob.glob(os.path.join(root_dir, 'image_0/*.png')))
        self.img2_paths = sorted(glob.glob(os.path.join(root_dir, 'image_1/*.png')))

        # Ground truth flow files (if any). Could be .flo, .pfm, or something else.
        self.flow_paths = sorted(glob.glob(os.path.join(root_dir, 'flow/*.flo')))

        # Basic sanity checks
        assert len(self.img1_paths) == len(self.img2_paths), "Mismatch in number of images."
        if len(self.flow_paths) > 0:
            assert len(self.img1_paths) == len(self.flow_paths), "Mismatch in flow files."

    def __len__(self):
        return len(self.img1_paths)

    def __getitem__(self, idx):
        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if len(self.flow_paths) > 0:
            flow_path = self.flow_paths[idx]
            flow = self.read_flo_file(flow_path)
        else:
            flow = None

        sample = {'img1': img1, 'img2': img2, 'flow': flow}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def read_flo_file(self, filename):
        """
        Reads a .flo file and returns a numpy array of shape [H, W, 2].
        Adjust as needed to match your flow file format.
        """
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, dtype=np.float32, count=1)
            if magic != 202021.25:
                raise Exception('Invalid .flo file')
            w = np.fromfile(f, dtype=np.int32, count=1).item()
            h = np.fromfile(f, dtype=np.int32, count=1).item()
            data = np.fromfile(f, dtype=np.float32, count=2 * w * h)
            flow = np.reshape(data, (h, w, 2))
        return flow


# ----------------------------------------------------------------------------------
# 2. Example Transform Class
# ----------------------------------------------------------------------------------

class FlowTransform(object):
    """
    Example transform that converts PIL images and flow to Torch tensors.
    You can add data augmentations here (random crops, flips, color jitter, etc.).
    """

    def __call__(self, sample):
        img1, img2, flow = sample['img1'], sample['img2'], sample['flow']

        to_tensor = T.ToTensor()
        img1_tensor = to_tensor(img1)
        img2_tensor = to_tensor(img2)

        if flow is not None:
            # flow should be a numpy array of shape (H, W, 2).
            flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        else:
            # Dummy flow if no ground truth is available.
            flow_tensor = torch.zeros(2, img1_tensor.shape[1], img1_tensor.shape[2])

        return {
            'img1': img1_tensor,
            'img2': img2_tensor,
            'flow': flow_tensor
        }


# ----------------------------------------------------------------------------------
# 3. FlowNetSimple Network
# ----------------------------------------------------------------------------------

class FlowNetSimple(nn.Module):
    """
    A simplified FlowNetSimple-based network.
    Input:  Two stacked RGB images => 6 channels
    Output: Optical flow => 2 channels
    """

    def __init__(self, input_channels=6):
        super(FlowNetSimple, self).__init__()

        # Encoder part
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        # Decoder part (simplified, no skip connections)
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # Flow prediction layer (2 channels for (u, v))
        self.predict_flow = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, img1, img2):
        """
        img1, img2: tensors of shape (B, 3, H, W)
        """
        x = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)

        # Encoder
        out_conv1 = self.relu(self.conv1(x))
        out_conv2 = self.relu(self.conv2(out_conv1))
        out_conv3 = self.relu(self.conv3(out_conv2))
        out_conv3_1 = self.relu(self.conv3_1(out_conv3))
        out_conv4 = self.relu(self.conv4(out_conv3_1))
        out_conv4_1 = self.relu(self.conv4_1(out_conv4))
        out_conv5 = self.relu(self.conv5(out_conv4_1))
        out_conv5_1 = self.relu(self.conv5_1(out_conv5))
        out_conv6 = self.relu(self.conv6(out_conv5_1))

        # Decoder (no skip connections here)
        out_deconv5 = self.relu(self.deconv5(out_conv6))
        out_deconv4 = self.relu(self.deconv4(out_deconv5))
        out_deconv3 = self.relu(self.deconv3(out_deconv4))
        out_deconv2 = self.relu(self.deconv2(out_deconv3))

        flow = self.predict_flow(out_deconv2)
        return flow


# ----------------------------------------------------------------------------------
# 4. EPE Loss Function
# ----------------------------------------------------------------------------------

def EPE_loss(pred_flow, true_flow, mask=None):
    """
    pred_flow: (B, 2, H, W)
    true_flow: (B, 2, H, W)
    mask: (B, 1, H, W) optional, if you want to focus on specific pixels
    Returns mean End-Point Error.
    """
    epe_map = torch.sqrt(torch.sum((pred_flow - true_flow) ** 2, dim=1))  # (B, H, W)

    if mask is not None:
        epe_map = epe_map * mask.squeeze(1)
        return epe_map.sum() / (mask.sum() + 1e-8)
    else:
        return epe_map.mean()


# ----------------------------------------------------------------------------------
# 5. Basic Training and Validation Loops
# ----------------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, sample in enumerate(dataloader):
        img1 = sample['img1'].to(device)
        img2 = sample['img2'].to(device)
        flow_gt = sample['flow'].to(device)

        optimizer.zero_grad()

        pred_flow = model(img1, img2)

        loss = EPE_loss(pred_flow, flow_gt)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            img1 = sample['img1'].to(device)
            img2 = sample['img2'].to(device)
            flow_gt = sample['flow'].to(device)

            pred_flow = model(img1, img2)
            loss = EPE_loss(pred_flow, flow_gt)
            val_loss += loss.item()

    return val_loss / len(dataloader)


# ----------------------------------------------------------------------------------
# 6. Example Main
# ----------------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = OpticalFlowDataset(
        root_dir="path/to/train",
        transform=FlowTransform()
    )
    val_dataset = OpticalFlowDataset(
        root_dir="path/to/val",
        transform=FlowTransform()
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = FlowNetSimple(input_channels=6).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "flownet_simple.pth")


if __name__ == "__main__":
    main()
