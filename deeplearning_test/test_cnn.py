import torch
import torch.nn as nn

from mnist_dataset import MNISTDataset
from torch.utils.data import DataLoader
from simple_cnn import SimpleCNN

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