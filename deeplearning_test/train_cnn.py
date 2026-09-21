import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_dataset import MNISTDataset
from simple_cnn import SimpleCNN



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
epochs = 5

for epoch in range(epochs):

    total_loss = 0
    correct = 0
    total = 0

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

        # 统计准确率
        predictions = output.argmax(dim=1)   # 得到预测值. tips: 取所有一维最大值，结果为一维数组[64]
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # 每100个 batch 打印一次
        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch + 1}, "
                f"Batch: {batch_idx}, "
                f"Loss: {loss.item():.4f}"
            )

    # 计算平均 Loss
    average_loss = total_loss / len(dataloader)

    # 计算准确率
    accuracy = correct / total
    print(
        f"Epoch {epoch + 1} 完成, "
        f"平均 Loss: {total_loss / len(dataloader):.4f}, "
        f"训练准确率: {accuracy * 100:.2f}%"
    )

# 保存模型参数
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型保存完成")