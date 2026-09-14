#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN 手写数字识别 (MNIST) - 低显存环境终极版
✅ 复用已修复的 AzureMNIST 路径逻辑
✅ 包含标准 CNN 模型定义、AMP混合精度训练及评估测试
✅ 完美适配 <2GB 小显存 GPU
"""

import os
import gc

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import autocast, GradScaler


# ======================
# 1. 终极修复版的数据集类
# ======================
class AzureMNIST(datasets.MNIST):
    AZURE_MIRROR = "https://azureopendatastorage.blob.core.windows.net/mnist/"

    # 分别定义训练集和测试集资源
    TRAIN_RESOURCES = [
        ("train-images-idx3-ubyte", "f68b316d79792418428c51bb8ef2967e"),
        ("train-labels-idx1-ubyte", "d53e105ee54ea40749a09fcbcd1e9432"),
    ]
    TEST_RESOURCES = [
        ("t10k-images-idx3-ubyte", "9269772db8dc8f23d15f01d0e5b321e2"),
        ("t10k-labels-idx1-ubyte", "68ec64f0535140ec5f7a10f9a2d445de")
    ]

    @property
    def raw_folder(self):
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", "processed")

    def _check_exists(self):
        # 根据当前加载的是训练集还是测试集，动态选择要检查的文件列表
        target_resources = self.TRAIN_RESOURCES if self.train else self.TEST_RESOURCES
        return all(
            os.path.exists(os.path.join(self.raw_folder, file_name))
            for file_name, _ in target_resources
        )


# ======================
# 2. 定义轻量级 CNN 模型
# ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )

        # 分类器层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平: 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 128),  # 全连接层
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout防止过拟合
            nn.Linear(128, 10)  # 输出10个类别(0-9)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================
# 3. 训练与测试函数 (含 AMP 与显存优化)
# ======================
def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, scaler):
    """单个Epoch的训练过程 (开启自动混合精度)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # ✅ 核心修改：使用 autocast 开启混合精度，大幅降低显存占用
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # ✅ 核心修改：使用 scaler 进行反向传播和权重更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch [{epoch}/10] | Batch [{batch_idx + 1}/{len(train_loader)}] | "
                  f"Loss: {running_loss / (batch_idx + 1):.4f} | Acc: {100. * correct / total:.2f}%")


def test_model(model, device, test_loader, criterion):
    """在测试集上评估模型 (关闭梯度计算以节省显存)"""
    model.eval()  # ✅ 关键1：将模型切换到评估模式（关闭 Dropout 等训练专属层）
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # ✅ 关键2：禁用梯度计算，大幅节省显存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\n🎯 【测试集结果】平均 Loss: {test_loss:.4f}, 准确率: {accuracy:.2f}%\n")
    return accuracy


def predict_my_digit(image_path, model, device):
    """
    加载自定义手写图片并进行预测
    """
    # 1. 读取图片并转为灰度图（'L'模式）
    image = Image.open(image_path).convert('L')

    # 2. 强制将图片尺寸调整为 28x28（防止尺寸不对报错）
    image = image.resize((28, 28))

    # 3. 将图片转为 Tensor 并归一化（必须和训练时的 transform 保持一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image)

    # 4. 增加一个“批次”维度，变成 (1, 1, 28, 28)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 5. 让模型进行预测（记得关闭梯度计算）
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probability, predicted_digit = torch.max(output.data, dim=1)

    # 6. 打印结果
    print(f"🖼️ 你手写的数字是：{predicted_digit.item()}")
    print(f"🎯 模型的置信度（概率）是：{probability.item():.4f}")


# ======================
# 4. 主执行入口
# ======================
if __name__ == "__main__":
    # 基础配置 (针对小显存进行了优化)
    BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 0.001

    # ⚠️ 请确保这里的绝对路径与你实际存放数据的目录一致
    BASE_DIR = "/tmp/pycharm_project_7b8685e5/data"

    # 自动选择设备 (优先GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 当前运行设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集和测试集
    print("▶ 正在加载 MNIST 数据集...")
    train_dataset = AzureMNIST(root=BASE_DIR, train=True, download=False, transform=transform)
    test_dataset = AzureMNIST(root=BASE_DIR, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数、优化器和 AMP Scaler
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # ← 新增：用于缩放梯度防止下溢

    print("=" * 60)
    print("▶ 开始训练 CNN 模型...")
    print("=" * 60)

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, scaler)
        test_model(model, device, test_loader, criterion)

        # ✅ 关键3：每个Epoch结束后定期清理显存碎片，防止 OOM
        gc.collect()
        torch.cuda.empty_cache()

    print("✅ 训练完成！")

    # 训练完成后，测试自己的手写图片
    # 注意：请确保当前目录下有 my_digit.png 这张图片
    test_image_path = "./数字2.PNG"
    if os.path.exists(test_image_path):
        predict_my_digit(test_image_path, model, device)
    else:
        print("⚠️ 未找到测试图片，请确保 .png 在当前目录下")