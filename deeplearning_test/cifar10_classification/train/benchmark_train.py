"""CIFAR-10 训练性能诊断：独立运行，不修改正式训练脚本或模型权重文件。"""
import sys
import time
from pathlib import Path

# 允许从 train/ 子目录直接运行本脚本
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn, optim
from datasets.cifar10_dataset import get_train_dataloader
from experiment_config import DEVICE, MODEL_NAME, NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY
from models.resnet18 import ResNet18
from models.simple_cnn import SimpleCNN

WARMUP_BATCHES = 3
TEST_BATCHES = 20
USE_CHANNELS_LAST = True

MODEL_CONFIG = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18,
}


def main():
    if MODEL_NAME not in MODEL_CONFIG:
        raise ValueError(f"不支持的模型名称：{MODEL_NAME}")

    loader = get_train_dataloader()
    model = MODEL_CONFIG[MODEL_NAME](num_classes=NUM_CLASSES).to(DEVICE)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=True,
    )

    print(f"模型：{MODEL_NAME} | 设备：{DEVICE} | Batch Size：{loader.batch_size}")
    print(f"PyTorch 计算线程数：{torch.get_num_threads()}")
    print(f"channels_last：{USE_CHANNELS_LAST} | 预热：{WARMUP_BATCHES} | 测试：{TEST_BATCHES}")

    iterator = iter(loader)
    times = {key: 0.0 for key in (
        "数据加载与增强", "数据传输/布局转换", "前向传播及Loss", "梯度清零", "反向传播", "优化器更新"
    )}

    def sync():
        # 目前用户在 CPU 上运行；保留同步以兼容未来 CUDA 测试
        if torch.device(DEVICE).type == "cuda":
            torch.cuda.synchronize()

    total_batches = WARMUP_BATCHES + TEST_BATCHES
    for i in range(total_batches):
        record = i >= WARMUP_BATCHES
        start = time.perf_counter()
        try:
            images, labels = next(iterator)
        except StopIteration:
            raise RuntimeError("数据集不足以完成本次测试") from None
        if record:
            times["数据加载与增强"] += time.perf_counter() - start

        start = time.perf_counter()
        if USE_CHANNELS_LAST:
            images = images.to(DEVICE, memory_format=torch.channels_last)
        else:
            images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        sync()
        if record:
            times["数据传输/布局转换"] += time.perf_counter() - start

        start = time.perf_counter()
        output = model(images)
        loss = criterion(output, labels)
        sync()
        if record:
            times["前向传播及Loss"] += time.perf_counter() - start

        start = time.perf_counter()
        optimizer.zero_grad()
        if record:
            times["梯度清零"] += time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        sync()
        if record:
            times["反向传播"] += time.perf_counter() - start

        start = time.perf_counter()
        optimizer.step()
        sync()
        if record:
            times["优化器更新"] += time.perf_counter() - start

        if (i + 1) % 5 == 0 or i + 1 == total_batches:
            print(f"已完成 {i + 1}/{total_batches} Batch")

    total = sum(times.values())
    print("\n========== 训练耗时诊断结果 ==========")
    for name, seconds in times.items():
        print(f"{name:<14} {seconds:8.2f} 秒  ({seconds / total * 100:5.1f}%)")
    print("--------------------------------------")
    print(f"测试总耗时：{total:.2f} 秒（不含预热）")
    print(f"平均每 Batch：{total / TEST_BATCHES:.3f} 秒")
    print("说明：只更新内存中的临时模型参数，不保存或覆盖任何 .pth 文件。")


if __name__ == "__main__":
    main()
