"""标准 ImageNet ResNet18；训练可初始化预训练权重，测试仅加载实验权重。"""
import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNet18(ResNet):
    def __init__(self, num_classes=10, pretrained=False, weights_path=None):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        if pretrained:
            if weights_path is None:
                raise ValueError("预训练初始化必须指定已校验的本地权重")
            self.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        # 主干继承预训练参数，分类层针对当前数据集重新初始化。
        self.fc = nn.Linear(self.fc.in_features, num_classes)

    def forward_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.forward_features(x))
