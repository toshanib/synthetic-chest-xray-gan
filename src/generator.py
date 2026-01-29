import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # Project latent vector into feature map
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)

        # Upsampling network
        self.net = nn.Sequential(
            # (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (128, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        """
        z: Tensor of shape (batch_size, latent_dim)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 256, 7, 7)
        img = self.net(x)
        return img
