import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os

# Import your custom dataset
from dataset import MovingObjectDisparityDataset

##############################################################################
# 1. DEFINE A SIMPLIFIED FLOWNET-SIMPLE MODEL
##############################################################################
class FlowNetSimple(nn.Module):
    """
    A mini version of FlowNetSimple that:
      - Stacks im0 and im1 along the channel dimension -> 6 input channels.
      - Uses a contractive encoder to shrink spatial size but expand feature depth.
      - Uses upconvolutions to expand back to a prediction of the same spatial size.
      - Predicts a 1-channel 'disparity' (or 2-channel flow in real FlowNet).
    """

    def __init__(self):
        super(FlowNetSimple, self).__init__()
        # Here we define only a few layers to keep it short.
        # The original FlowNetSimple from the paper is deeper.
        # Input = 6 channels (RGB im0 + RGB im1)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),  # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),# downsample
            nn.ReLU(inplace=True)
        )

        # A simple upconvolution path back to ~1/2 original scale
        # Real FlowNet would do multiple upconvs + skip connections
        self.decoder_upconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output layer (assuming final size is 1/2 the input dimension).
        # If you need the exact original W,H, do another upconv or simply
        # upsample with interpolation.
        self.predict_disparity = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, im0, im1):
        """
        Inputs:
          im0, im1 are [B,3,H,W] each
        Return:
          disp of shape [B,1,H/2,W/2] in this simplified version.
        """
        # Stack along channel dimension
        x = torch.cat((im0, im1), dim=1)  # shape = [B,6,H,W]

        # Encode
        x = self.encoder(x)               # shape roughly [B,256,H/8,W/8]

        # Decode
        x = self.decoder_upconv(x)        # shape roughly [B,64,H/2,W/2]

        # Predict 1-channel disparity
        disp = self.predict_disparity(x)  # shape [B,1,H/2,W/2]
        return disp

##############################################################################
# 2. TRAINING LOOP
##############################################################################
def train_flownet_simple():
    # Basic hyperparameters
    batch_size = 2
    num_epochs = 10
    learning_rate = 1e-4

    # Paths
    dataset_root = 'dataset'

    # Create dataset and loader
    train_dataset = MovingObjectDisparityDataset(root_dir=dataset_root,
                                                 use_disp_name='disp0')
    print("Length of dataset:", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

    # Instantiate the model
    model = FlowNetSimple().cuda()

    # Define an optimizer and a simple L1 loss (or MSE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    # Training
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            im0 = sample['im0'].cuda()   # [B,3,H,W]
            im1 = sample['im1'].cuda()   # [B,3,H,W]
            disp_gt = sample['disp'].cuda()  # [B,1,H,W] or [B,H,W]

            # Forward
            disp_pred = model(im0, im1)  # [B,1,H/2,W/2] in this simplified example

            # You may need to downsample disp_gt to match the shape:
            _, _, predH, predW = disp_pred.shape
            disp_gt_resized = torch.nn.functional.interpolate(disp_gt, size=(predH, predW),
                                                              mode='bilinear', align_corners=False)

            loss = criterion(disp_pred, disp_gt_resized)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/flownet_simple.pth')
    print("Model saved to checkpoints/flownet_simple.pth.")

if __name__ == '__main__':
    train_flownet_simple()
