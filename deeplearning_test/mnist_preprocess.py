import torch
from torchvision import datasets, transforms

""""
实验第一步：先将MNIST图片下载下来预处理
tips: 从https://github.com/fgnt/mnist下载4个训练集.gz文件后，在windows环境下使用rar解压会改变文件名称，
'train-images-idx3-ubyte.gz'->'train-images.idx3-ubyte'，实际需要'train-images-idx3-ubyte'
"""

import os

print("当前运行目录:")
print(os.getcwd())

print("\ndata目录:")
print(os.path.abspath("../data"))

print("\nMNIST目录:")
print(os.path.abspath("../data/MNIST"))

print("\nraw目录:")
print(os.path.abspath("../data/MNIST/raw"))

print("\n文件:")
print(os.listdir("../data/MNIST/raw"))


raw_path = "../data/MNIST/raw"

for file in os.listdir(raw_path):
    path = os.path.join(raw_path, file)
    print(file, os.path.getsize(path))


root="../data/MNIST"

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])


# 下载训练集
train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=False,
    transform=transform
)


# 下载测试集
test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=False,
    transform=transform
)


print("训练集数量:", len(train_dataset))
print("测试集数量:", len(test_dataset))