from torch import nn


class Darknet53Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.af = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class Darknet53Residual(nn.Module):
    def __init__(self, io_channels, mid_channels):
        super().__init__()
        self.conv1 = Darknet53Conv(io_channels, mid_channels, 1, 1, 0)
        self.conv2 = Darknet53Conv(mid_channels, io_channels, 3, 1, 1)

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
        self.conv1 = Darknet53Conv(3, 32, 3, 1, 1)
        self.conv2 = Darknet53Conv(32, 64, 3, 2, 1)  # 416x416 -> 208x208
        self.residuals1 = Darknet53Residuals(64, 32, 1)
        self.conv3 = Darknet53Conv(64, 128, 3, 2, 1)  # 208x208 -> 104x104
        self.residuals2 = Darknet53Residuals(128, 64, 2)
        self.conv4 = Darknet53Conv(128, 256, 3, 2, 1)  # 104x104 -> 52x52
        self.residuals3 = Darknet53Residuals(256, 128, 8)
        self.conv5 = Darknet53Conv(256, 512, 3, 2, 1)  # 52x52 -> 26x26
        self.residuals4 = Darknet53Residuals(512, 256, 8)
        self.conv6 = Darknet53Conv(512, 1024, 3, 2, 1)  # 26x26 -> 13x13
        self.residuals5 = Darknet53Residuals(1024, 512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residuals1(x)
        x = self.conv3(x)
        x = self.residuals2(x)
        x = self.conv4(x)
        size1 = self.residuals3(x)
        x = self.conv5(x)
        size12 = self.residuals4(x)
        x = self.conv6(x)
        size13 = self.residuals5(x)
        return size11, size12, size13

class Darknet53Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Darknet53Conv(3, 32, 3, 1, 1)
        self.conv2 = Darknet53Conv(32, 64, 3, 2, 1)  # 416x416 -> 208x208
        self.residuals1 = Darknet53Residuals(64, 32, 1)
        self.conv3 = Darknet53Conv(64, 128, 3, 2, 1)  # 208x208 -> 104x104
        self.residuals2 = Darknet53Residuals(128, 64, 2)
        self.conv4 = Darknet53Conv(128, 256, 3, 2, 1)  # 104x104 -> 52x52
        self.residuals3 = Darknet53Residuals(256, 128, 8)
        self.conv5 = Darknet53Conv(256, 512, 3, 2, 1)  # 52x52 -> 26x26
        self.residuals4 = Darknet53Residuals(512, 256, 8)
        self.conv6 = Darknet53Conv(512, 1024, 3, 2, 1)  # 26x26 -> 13x13
        self.residuals5 = Darknet53Residuals(1024, 512, 4)
        
        self.avgpool = nn.AvgPool2d(13) # 13x13 -> 1x1
        self.conv2d = nn.Conv2d(1024, 1000, 1, 1, 0)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residuals1(x)
        x = self.conv3(x)
        x = self.residuals2(x)
        x = self.conv4(x)
        x = self.residuals3(x)
        x = self.conv5(x)
        x = self.residuals4(x)
        x = self.conv6(x)
        x = self.residuals5(x)
        x = self.conv2d(x)
        x = self.softmax(x)
        return x
