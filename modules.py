import torch
from torch import nn, cat
import torch.nn.functional as F


class Darknet53Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.af = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class Darknet53Residual(nn.Module):
    def __init__(self, io_channels, mid_channels):
        super().__init__()
        self.conv1 = Darknet53Conv(io_channels, mid_channels, 1)
        self.conv2 = Darknet53Conv(mid_channels, io_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = residual + x
        return x


class Darknet53Residuals(nn.Module):
    def __init__(self, io_channels, mid_channels, n_times):
        super().__init__()
        self.residuals = nn.Sequential(
            *[Darknet53Residual(io_channels, mid_channels) for _ in range(n_times)]
        )

    def forward(self, x):
        x = self.residuals(x)
        return x


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Darknet53Conv(3, 32)
        self.conv2 = Darknet53Conv(32, 64, 3, 2)  # 416x416 -> 208x208
        self.residuals1 = Darknet53Residuals(64, 32, 1)
        self.conv3 = Darknet53Conv(64, 128, 3, 2)  # 208x208 -> 104x104
        self.residuals2 = Darknet53Residuals(128, 64, 2)
        self.conv4 = Darknet53Conv(128, 256, 3, 2)  # 104x104 -> 52x52
        self.residuals3 = Darknet53Residuals(256, 128, 8)
        self.conv5 = Darknet53Conv(256, 512, 3, 2)  # 52x52 -> 26x26
        self.residuals4 = Darknet53Residuals(512, 256, 8)
        self.conv6 = Darknet53Conv(512, 1024, 3, 2)  # 26x26 -> 13x13
        self.residuals5 = Darknet53Residuals(1024, 512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residuals1(x)
        x = self.conv3(x)
        x = self.residuals2(x)
        x = self.conv4(x)
        x = self.residuals3(x)
        x52 = x
        x = self.conv5(x)
        x = self.residuals4(x)
        x26 = x
        x = self.conv6(x)
        x = self.residuals5(x)
        x13 = x
        return x52, x26, x13


class FeaturePyramidConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_chahnnels = out_channels * 2
        self.conv1 = Darknet53Conv(in_channels, out_channels, 1)
        self.conv2 = Darknet53Conv(out_channels, mid_chahnnels)
        self.conv3 = Darknet53Conv(mid_chahnnels, out_channels, 1)
        self.conv4 = Darknet53Conv(out_channels, mid_chahnnels)
        self.conv5 = Darknet53Conv(mid_chahnnels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class YOLOv3Head(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        mid_channels = in_channels * 2
        self.conv1 = Darknet53Conv(in_channels, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, (5 + num_classes) * 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FeaturePyramidUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Darknet53Conv(in_channels, in_channels // 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x


class FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()  # 13x13
        self.conv1 = FeaturePyramidConv(1024, 512)
        self.upsample1 = FeaturePyramidUpsample(512)  # 26x26
        self.conv2 = FeaturePyramidConv(768, 256)
        self.upsample2 = FeaturePyramidUpsample(256)  # 52x52
        self.conv3 = FeaturePyramidConv(384, 128)

    def forward(self, x):
        (x52, x26, x13) = x
        x = x13
        x = self.conv1(x)
        x13 = x
        x = self.upsample1(x)
        x = cat((x, x26), dim=1)
        x = self.conv2(x)
        x26 = x
        x = self.upsample2(x)
        x = cat((x, x52), dim=1)
        x = self.conv3(x)
        x52 = x
        return x52, x26, x13


class YOLOv3(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = Darknet53()
        self.neck = FeaturePyramid()
        self.head1 = YOLOv3Head(512, num_classes)
        self.head2 = YOLOv3Head(256, num_classes)
        self.head3 = YOLOv3Head(128, num_classes)

    def forward(self, x):
        (x52, x26, x13) = self.backbone(x)
        (x52, x26, x13) = self.neck((x52, x26, x13))
        x13 = self.head1(x13)
        x26 = self.head2(x26)
        x52 = self.head3(x52)
        return x52, x26, x13
