import torch
from torch import nn

# 卷积层相关参数：
#
# weight.shape = out_channels x in_channels x kernel_h x kernel_w
# bais.shap = out_channels

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层卷积CNN：3x32x32 -> 16x32x32
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,    # 输出的
            kernel_size=3,
            stride=1,
            padding=1)

        # 第二层卷积CNN：16x16x16 -> 32x16x16
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 第三层卷积CNN：32x8x8 -> 64x8x8
        # tips: weight为64x32x3x3，bias一个通道一个bias即64个bias
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # BatchNorm2d：处理CNN内部的特征图。每个outChannel都会计算均值和方差，并且有γ和β两个可学习值
        # y = γX + β, γ为weight，β为bias
        # 此外，batchNorm2d中会维护batch_mean、batch_var用于训练，running_mean、running_var用于推理
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # 最大池化
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接
        self.fc = nn.Linear(64 * 4 * 4, 10)

        # dropout
        self.dropout = nn.Dropout(p=0.5)

    # x: 参数通常是一批照片, [64, 3, 32, 32]
    def forward(self, x):
        # 1: 卷积 -> relu激活 -> 池化
        x = self.conv1(x)   # 3x32x32 -> 16x32x32
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)    # 16x32x32 -> 16x16x16

        # 2: 卷积 -> relu激活 -> 池化
        x = self.conv2(x)   # 16x16x16 -> 32x16x16
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)    # 32x16x16 -> 32x8x8

        # 3: 卷积 -> relu激活 -> 池化
        x = self.conv3(x)  # 32x8x8 -> 64x8x8
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)  # 64x8x8 -> 64x4x4

        # 展平
        x = x.view(x.size(0), -1)

        # dropout，随机丢弃特征
        x = self.dropout(x)
        # 全连接
        x = self.fc(x)

        return x