from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# 训练集随机打乱，测试集不打乱

def get_train_dataset():
    train_dataset = CIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()    # transform PIL Image to tensor, and 0~255 pixel to 0~1.0
    )

    return train_dataset

def get_test_dataset():
    test_dataset = CIFAR10(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return test_dataset

def get_train_dataloader():
    train_dataset = get_train_dataset()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True    # 随机打乱
    )

    return train_dataloader

def get_test_dataloader():
    test_dataset = get_test_dataset()

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    return test_dataloader