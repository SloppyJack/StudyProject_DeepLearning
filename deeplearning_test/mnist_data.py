from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from deeplearning_test.mnist_dataset import MNISTDataset

BASE_DIR = Path(__file__).resolve().parent.parent
MNIST_DIR = BASE_DIR / "data" / "MNIST" / "raw"

# 创建训练集
def get_train_dataset():
    return MNISTDataset(
        image_path = MNIST_DIR / "train-images-idx3-ubyte",
        label_path = MNIST_DIR / "train-labels-idx1-ubyte"
    )


# 创建测试集
def get_test_dataset():
    return MNISTDataset(
        image_path=MNIST_DIR / "t10k-images-idx3-ubyte",
        label_path=MNIST_DIR / "t10k-labels-idx1-ubyte"
    )


# 创建训练 DataLoader
def get_train_dataloader(batch_size=64):
    dataset = get_train_dataset()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


# 创建测试 DataLoader
def get_test_dataloader(batch_size=64):
    dataset = get_test_dataset()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

#
def get_train_val_dataset():
    dataset = get_train_dataset()

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [54000, 6000],
        generator=generator
    )

    return train_dataset, val_dataset

# 分成训练集和验证集
def get_train_val_dataloader(batch_size=64):
    train_dataset, val_dataset = get_train_val_dataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader