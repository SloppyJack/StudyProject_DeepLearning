import torch
import torch.nn as nn
from torch import optim

from datasets.cifar10_dataset import get_train_dataloader, get_test_dataloader
from models.simple_cnn import SimpleCNN

train_dataloader = get_train_dataloader()
test_dataloader = get_test_dataloader()

model = SimpleCNN()

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 15

best_accuracy = 0.0

for epoch in range(epochs):
    # 训练模型
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_dataloader):
        # 前向传播
        output = model(images)
        # 计算 Loss
        loss = criterion(output, labels)
        # 清空旧梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 统计total_loss
        total_loss += loss.item()

        # 统计准确率
        predicted = output.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = total_loss / len(train_dataloader)
    train_accuracy = correct / total

    # 测试模型
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            # 前向传播
            output = model(images)
            # 计算loss
            loss = criterion(output, labels)
            test_loss += loss.item()
            # 计算测试正确率
            predicted = output.argmax(dim=1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_dataloader)
    test_accuracy = test_correct / test_total

    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            "../outputs/checkpoints/best_model.pth"
        )

        print(
            f"保存最佳模型，"
            f"Test Accuracy: {best_accuracy * 100:.2f}%"
        )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy * 100:.2f}% "
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_accuracy * 100:.2f}%"
    )