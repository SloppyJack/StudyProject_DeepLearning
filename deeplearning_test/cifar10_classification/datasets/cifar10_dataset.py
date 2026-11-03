import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from experiment_config import DATA_DIR, BATCH_SIZE, CIFAR10_MEAN, CIFAR10_STD

# 只生成一次随机索引
indices = torch.randperm(
    50000,
    generator=torch.Generator().manual_seed(42)
).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 32x32 -> 40x40，随机剪裁成32x32
    transforms.RandomHorizontalFlip(p=0.5), # 读取一张图片，50%概率水平翻转

    transforms.ToTensor(),  # 将原始图片转换成3x32x32，便于后面交给CNN训练。此外还将像素值转为0~1
    transforms.Normalize(
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD
    )
])

def _get_full_train_dataset(transform):
    return CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

# 训练集随机打乱，测试集不打乱
def get_train_dataset():
    dataset = _get_full_train_dataset(train_transform)
    return Subset(dataset, train_indices)

def get_val_dataset():
    dataset = _get_full_train_dataset(test_transform)
    return Subset(dataset, val_indices)

def get_test_dataset():
    test_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform
    )

    return test_dataset

def get_train_dataloader():
    train_dataset = get_train_dataset()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True    # 随机打乱
    )

    return train_dataloader

def get_val_dataloader():
    val_dataset = get_val_dataset()

    return DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def get_test_dataloader():
    test_dataset = get_test_dataset()

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_dataloader