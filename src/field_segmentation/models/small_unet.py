"""Small U-Net baseline."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Helper module for SmallUNet that performs two consecutive convolutional layers with ReLU activation and dropout.
    """
    def __init__(self, in_channels: int, out_channels):
        """
        Initialize DoubleConv module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DoubleConv.
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Small U-Net architecture for image segmentation.
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 2):
        """
        Initialize UNet model.
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            n_classes (int): Number of output classes for segmentation.
        """
        super().__init__()

        base_channels = 16
        c1 = base_channels
        c2 = 2 * base_channels
        c3 = 4 * base_channels
        c4 = 8 * base_channels
        c5 = 16 * base_channels

        # Encoder
        self.conv1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv5 = DoubleConv(c4, c5)

        # Decoder
        self.upconv6 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(c4 + c4, c4)

        self.upconv7 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(c3 + c3, c3)

        self.upconv8 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(c2 + c2, c2)

        self.upconv9 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(c1 + c1, c1)

        # Output
        self.out = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SmallUNet.
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output segmentation map of shape [B, n_classes, H, W].
        """
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        # Middle
        conv5 = self.conv5(pool4)

        # Decoder
        up6 = self.upconv6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(up6)

        up7 = self.upconv7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(up7)

        up8 = self.upconv8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(up8)

        up9 = self.upconv9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(up9)

        outputs = self.out(conv9)

        return outputs