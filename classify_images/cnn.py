import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # Remove inplace operation from ReLU
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # Store intermediate results to avoid in-place operations
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels, out_channels, stride=stride, dilation=dilation
        )
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, dilation=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        # Remove inplace operation from ReLU
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # Store intermediate results to avoid in-place operations
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # Changed from += to +
        out = self.relu(out)
        return out


class MushroomClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial = nn.Sequential(
            ConvBlock(3, 32, stride=1), ConvBlock(32, 32, stride=2)
        )

        self.block1 = ResBlock(32, 64, stride=2, dilation=2)
        self.block2 = ResBlock(64, 128, stride=2, dilation=2)
        self.block3 = ResBlock(128, 256, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Store intermediate results to avoid in-place operations
        out = self.initial(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        # out = self.dropout(out)
        out = self.fc(out)

        return out
