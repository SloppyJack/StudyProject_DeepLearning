import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cnn_digit_recog.mnist_data import get_train_val_dataloader, get_test_dataloader
from simple_cnn import SimpleCNN



# =========================
# 2. 准备数据
# =========================

train_dataloader, val_dataloader = get_train_val_dataloader()

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

# 保存每个 Epoch 的平均 Loss
loss_history = []
# 保存每个 Epoch 的训练准确率
accuracy_history = []
# 验证集准确率
val_accuracy_history = []

best_val_accuracy = 0.0

torch.save(
    model.state_dict(),
    "pth/model_epoch_0.pth"
)

for epoch in range(epochs):

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_dataloader):

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

        # # 每100个 batch 打印一次
        #     if batch_idx % 100 == 0:
        #         print(
        #             f"Epoch: {epoch + 1}, "
        #             f"Batch: {batch_idx}, "
        #             f"Loss: {loss.item():.4f}"
        #         )

    # 计算平均 Loss
    average_loss = total_loss / len(train_dataloader)
    # 保存当前 Epoch 的平均 Loss
    loss_history.append(average_loss)

    # 计算准确率
    accuracy = correct / total
    accuracy_history.append(accuracy)
    print(
        f"Epoch {epoch + 1} 完成, "
        f"平均 Loss: {total_loss / len(train_dataloader):.4f}, "
        f"训练准确率: {accuracy * 100:.2f}%"
    )

    # 测试当前epoch的模型
    val_correct =0
    val_total = 0

    # 切换到推理模式
    model.eval()

    with torch.no_grad():

        for images, labels in val_dataloader:
            # 前向传播
            output = model(images)

            # 找出预测数字
            predictions = output.argmax(dim=1)

            # 当前 batch 有多少张图片
            val_total += labels.size(0)

            # 当前 batch 预测正确多少张
            val_correct += (predictions == labels).sum().item()

    val_accuracy = val_correct / val_total
    val_accuracy_history.append(val_accuracy)

    print(
        f"验证集准确率: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "pth/mnist_cnn.pth"
        )

        print(
            f"发现更好的模型，已保存，"
            f"最佳验证集准确率: {best_val_accuracy * 100:.2f}%"
        )

    torch.save(
        model.state_dict(),
        f"pth/model_epoch_{epoch + 1}.pth"
    )

    # 切换到训练模式
    model.train()

print("每个 Epoch 的 Loss：", loss_history)
print("每个 Epoch 的 Accuracy: ", accuracy_history)
print("每个 Epoch 的验证准确率:", val_accuracy_history)


# 加载验证集表现最好的模型
model.load_state_dict(
    torch.load("pth/mnist_cnn.pth")
)

# 开始推理
model.eval()

test_dataloader = get_test_dataloader()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_dataloader:

        output = model(images)

        predictions = output.argmax(dim=1)

        test_total += labels.size(0)
        test_correct += (predictions == labels).sum().item()

test_accuracy = test_correct / test_total

print(
    f"最终测试集准确率: {test_accuracy * 100:.2f}%"
)

# 绘制 Loss 曲线
epoch_list = range(1, epochs + 1)
plt.plot(epoch_list, loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 绘制训练集和测试集 Accuracy 曲线
plt.figure()

plt.plot(
    epoch_list,
    accuracy_history,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epoch_list,
    val_accuracy_history,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()