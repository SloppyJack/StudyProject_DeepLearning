import torch
import torch.nn as nn

from mnist_dataset import MNISTDataset
from torch.utils.data import DataLoader


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
            kernel_size=2,  # 使用2x2窗口
            stride=2    # 移动步长为2
        )

        # 全连接层：每个特征都和输出相连
        self.fc = nn.Linear(
            32 * 7 * 7,
            10
        )

    # 前向传播
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
            x.size(0),  # x.size(0)取的第0维大小，即一批图片的数量
            -1
        )

        # 全连接
        x = self.fc(x)

        return x

# 创建模型
model = SimpleCNN()

# 训练数据集
dataset = MNISTDataset(
    "../data/MNIST/raw/train-images-idx3-ubyte",
    "../data/MNIST/raw/train-labels-idx1-ubyte"
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True    # 确保每个epoch训练数据随机打乱（Epoch = 把整个训练集完整训练一遍）
)
images, labels = next(iter(dataloader))

output = model(images)
print("输入尺寸:", images.shape)
print("输出尺寸:", output.shape)
print("第一张图片的输出:", output[0])

# 创建损失函数
criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels)
print("Loss:", loss.item())

loss.backward()

print("conv1 第一个卷积核的梯度:")
print(model.conv1.weight.grad[0])

old_weight = model.conv1.weight.data[0].clone()

# SGD: Stochastic Gradient Descent，随机梯度下降。w_new = w_old - lr*grad
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1  # learning_rate，
)
optimizer.step()

new_weight = model.conv1.weight.data[0]

print("更新前:")
print(old_weight)

print("更新后:")
print(new_weight)

# 清空旧梯度
#    ↓
# 前向传播
#    ↓
# 计算 Loss
#    ↓
# 反向传播
#    ↓
# 更新参数