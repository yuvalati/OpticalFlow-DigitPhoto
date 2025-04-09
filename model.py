import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import our custom dataset
from dataset import FlowDataset


class FlowNetSimple(nn.Module):
    """
    A simplified FlowNetSimple architecture:
      - We stack im0 and im1 along the channel axis: input shape = (6,H,W).
      - Then do a series of conv layers with stride 2 in some of them.
      - Finally predict a 1-channel or 2-channel output (here 1 for 'disp0').
    """

    def __init__(self, output_channels=1):
        super().__init__()
        # output_channels=1 if you only want a single disparity channel,
        # or =2 if you want 2D optical flow.

        # Convolutional encoder part
        # (kernel_size, stride, padding) follows the original FlowNet style
        # but simplified a bit.

        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # A simple "head" that outputs our final flow/disparity
        self.predict_flow = nn.Conv2d(512, output_channels, kernel_size=3, stride=1, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, im0, im1):
        """
        im0, im1: [B,3,H,W] each
        We'll stack them to get input shape [B,6,H,W].
        Returns a single-channel or 2-channel predicted flow map [B, outC, H/32, W/32].
        """
        x = torch.cat([im0, im1], dim=1)  # shape [B, 6, H, W]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        flow = self.predict_flow(x)
        return flow


def train_one_epoch(model, dataloader, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0

    for batch_idx, sample in enumerate(dataloader):
        im0 = sample['im0'].to(device)  # [B,3,H,W]
        im1 = sample['im1'].to(device)  # [B,3,H,W]
        disp0_gt = sample['disp0'].to(device)  # [B,1,H,W] (our "ground truth")

        # Forward
        pred = model(im0, im1)
        # pred shape is [B,1,H_out,W_out], typically smaller in spatial dims than input.
        # If your net downsamples by factor 32, you may want to downsample disp0_gt to match it:
        B, _, H_out, W_out = pred.shape
        disp0_down = F.interpolate(disp0_gt, size=(H_out, W_out), mode='nearest')

        # Example L1 loss:
        loss = F.l1_loss(pred, disp0_down)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Create dataset & loader
    dataset_path = './Dataset'
    train_ds = FlowDataset(root_dir=dataset_path)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

    # 2) Create model & optimizer
    model = FlowNetSimple(output_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3) Train loop
    num_epochs = 5
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Done. You can now do further validation or save your model.
    # e.g. torch.save(model.state_dict(), 'flownet_simple.pth')


if __name__ == '__main__':
    main()

