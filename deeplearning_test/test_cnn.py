import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 第一层卷积：1x28x28 -> 16x28x28
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 输入图像通道数，原始灰度图片通道数为1
            out_channels=16,    # 使用16个卷积核观察图片
            kernel_size=3,  # 每次用3x3小窗口看图片
            padding=1   # 给图片周围补一圈0，图片变成30x30，卷积后结果依然为28x28
        )

        # 第二层卷积：16x28x28 -> 32x28x28
        self.conv2 = nn.Conv2d(
            in_channels=16, # 16个特征图有16个通道
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # 最大池化
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 全连接层
        self.fc = nn.Linear(
            32 * 7 * 7,
            10
        )


    def forward(self, x):

        # 第一层卷积
        x = self.conv1(x)

        # 激活函数
        x = torch.relu(x)

        # 池化
        x = self.pool(x)


        # 第二层卷积
        x = self.conv2(x)

        # 激活函数
        x = torch.relu(x)

        # 池化
        x = self.pool(x)


        # 展平
        x = x.view(
            x.size(0),
            -1
        )


        # 全连接
        x = self.fc(x)

        return x

    # 创建模型
model = SimpleCNN()

print(model)