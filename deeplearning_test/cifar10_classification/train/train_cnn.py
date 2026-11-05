import time

import torch
import torch.nn as nn
from torch import optim

from datasets.cifar10_dataset import get_train_dataloader, get_val_dataloader
from experiment_config import NUM_CLASSES, CHECKPOINT_DIR, DEVICE, MODEL_NAME, LEARNING_RATE, WEIGHT_DECAY, STEP_SIZE, \
    GAMMA, EPOCHS, PATIENCE, BATCH_SIZE, SCHEDULER_T_MAX, SCHEDULER_ETA_MIN
from models.resnet18 import ResNet18
from models.simple_cnn import SimpleCNN
from experiment_config import USE_MIXUP, MIXUP_ALPHA, MIXUP_EPOCHS
from utils.mixup import mixup_batch
from experiment_config import USE_CATDOG_AUX, CATDOG_AUX_WEIGHT
from utils.catdog_aux import catdog_aux_loss
from experiment_config import ONECYCLE_PCT_START, ONECYCLE_DIV_FACTOR, ONECYCLE_FINAL_DIV_FACTOR

BEST_MODEL_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"

train_dataloader = get_train_dataloader()
val_dataloader = get_val_dataloader()

# 模型配置
MODEL_CONFIG = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18
}
if MODEL_NAME not in MODEL_CONFIG:
    raise ValueError(f"不支持的模型名称: {MODEL_NAME}")

# 创建模型
model = MODEL_CONFIG[MODEL_NAME](num_classes=NUM_CLASSES)
model = model.to(DEVICE, memory_format=torch.channels_last)

# Training-only auxiliary head; the saved ten-class model remains compatible.
aux_head = None
parameters = list(model.parameters())
if USE_CATDOG_AUX:
    if MODEL_NAME != "resnet18":
        raise ValueError("Cat/dog auxiliary task currently requires ResNet18")
    aux_head = nn.Linear(model.fc.in_features, 2).to(DEVICE)
    parameters += list(aux_head.parameters())

print(f"当前模型：{MODEL_NAME}")
print(f"运行设备：{DEVICE}")
print(f"训练轮数：{EPOCHS}")
print(f"初始学习率：{LEARNING_RATE}")
print(f"Batch Size：{BATCH_SIZE}")

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    parameters,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    foreach=True
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dataloader),
    pct_start=ONECYCLE_PCT_START,
    div_factor=ONECYCLE_DIV_FACTOR,
    final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
    cycle_momentum=False
)

epochs = EPOCHS

best_val_loss = float("inf")
counter = 0
patience = PATIENCE

for epoch in range(epochs):
    epoch_start = time.perf_counter()
    # 记录当前 epoch 使用的学习率
    current_lr = optimizer.param_groups[0]["lr"]

    # 训练模型
    model.train()

    train_start = time.perf_counter()
    train_total_loss = 0.0
    train_correct = 0
    train_total = 0
    aux_loss_sum = 0.0

    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)
        labels_a, labels_b, lam = labels, labels, 1.0
        if USE_MIXUP and epoch < MIXUP_EPOCHS:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)
        # 前向传播
        if aux_head is not None:
            features = model.forward_features(images)
            output = model.fc(features)
        else:
            output = model(images)
        # 计算 Loss
        loss = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)
        if aux_head is not None:
            auxiliary_loss = catdog_aux_loss(aux_head(features), labels_a, labels_b, lam)
            loss = loss + CATDOG_AUX_WEIGHT * auxiliary_loss
            aux_loss_sum += auxiliary_loss.item() * labels.size(0)
        # 清空旧梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        scheduler.step()

        # 统计total_loss
        train_total_loss += loss.item() * labels.size(0)

        # 统计准确率
        predicted = output.argmax(dim=1)
        train_total += labels.size(0)
        train_correct += (
            lam * (predicted == labels_a).sum().item()
            + (1 - lam) * (predicted == labels_b).sum().item()
        )

    train_time = time.perf_counter() - train_start
    train_loss = train_total_loss / train_total
    train_accuracy = train_correct / train_total

    # 验证模型
    model.eval()

    val_start = time.perf_counter()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(DEVICE, memory_format=torch.channels_last)
            labels = labels.to(DEVICE)
            # 前向传播
            output = model(images)
            # 计算loss
            loss = criterion(output, labels)
            val_loss_sum += loss.item() * labels.size(0)
            # 计算测试正确率
            predicted = output.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_time = time.perf_counter() - val_start
    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(),BEST_MODEL_PATH)
        if aux_head is not None:
            torch.save(aux_head.state_dict(), CHECKPOINT_DIR / f"{MODEL_NAME}_catdog_aux.pth")

        print(
            f"保存最佳模型，"
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )
    else:
        counter += 1

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy * 100:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    print(f"Epoch [{epoch + 1}/{epochs}], lr={current_lr:.6f}")
    if aux_head is not None:
        print(f"Aux Loss (batch mean): {aux_loss_sum / train_total:.6f}")

    if counter >= patience:
        print("Early Stopping!")
        break

    # OneCycleLR 在每次参数更新后调整学习率。
    epoch_time = time.perf_counter() - epoch_start
    print(f"训练耗时：{train_time:.2f} 秒")
    print(f"验证耗时：{val_time:.2f} 秒")
    print(f"Epoch 总耗时：{epoch_time:.2f} 秒")
