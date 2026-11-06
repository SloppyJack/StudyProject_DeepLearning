import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False  # 卷积层后面有BatchNorm，卷积偏置不必要
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False  # 卷积层后面有BatchNorm，卷积偏置不必要
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut：处理输入输出形状不一致的情况
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity() # 恒等映射

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


"""
ResNet18：1 + 4x2x2 + 1
初始1个卷积层conv1
+ 4个残差阶段，每个残差阶段2个BasicBlock，每个BasicBlock有2个卷积，4x2x2
+ 最后全连接层
"""
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        # 初始卷积：适配 CIFAR-10 的 32×32 图像
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个残差阶段
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 全局平均池化：显著减少全连接层参数，防止过拟合。batchx512x4x4 -> batchx512x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        自动生成残差层
        :param num_blocks: 残差块数量
        """
        layers = []

        layers.append(
            BasicBlock(self.in_channels, out_channels, stride)
        )
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(self.in_channels, out_channels, stride=1)
            )
        # nn.Sequential()用于组合多个残差层
        return nn.Sequential(*layers)

    def forward_features(self, x):
        """提取主分类器特征，供前向传播和只读可视化复用，不包含辅助头。"""
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self.fc(self.forward_features(x))

if __name__ == "__main__":
    model = ResNet18(num_classes=10)

    # 模拟一个批次的 CIFAR-10 图片
    x = torch.randn(2, 3, 32, 32)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print("模型参数量：", total_params)
