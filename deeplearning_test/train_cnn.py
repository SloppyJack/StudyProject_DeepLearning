import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_dataset import MNISTDataset


# =========================
# 1. 定义 CNN
# =========================
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(
            32 * 7 * 7,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# =========================
# 2. 准备数据
# =========================
dataset = MNISTDataset(
    "../data/MNIST/raw/train-images-idx3-ubyte",
    "../data/MNIST/raw/train-labels-idx1-ubyte"
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)


# =========================
# 3. 创建模型
# =========================
model = SimpleCNN()


# =========================
# 4. 创建 Loss
# =========================
criterion = nn.CrossEntropyLoss()


# =========================
# 5. 创建优化器
# =========================
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


# =========================
# 6. 开始训练
# =========================
epochs = 1

for epoch in range(epochs):

    total_loss = 0

    for batch_idx, (images, labels) in enumerate(dataloader):

        # ① 清空旧梯度
        optimizer.zero_grad()

        # ② 前向传播
        output = model(images)

        # ③ 计算 Loss
        loss = criterion(output, labels)

        # ④ 反向传播
        loss.backward()

        # ⑤ 更新参数
        optimizer.step()

        total_loss += loss.item()

        # 每100个 batch 打印一次
        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch + 1}, "
                f"Batch: {batch_idx}, "
                f"Loss: {loss.item():.4f}"
            )

    print(
        f"Epoch {epoch + 1} 完成, "
        f"平均 Loss: {total_loss / len(dataloader):.4f}"
    )