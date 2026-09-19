from mnist_dataset import MNISTDataset
from torch.utils.data import DataLoader


# 训练数据集
dataset = MNISTDataset(
    "../data/MNIST/raw/train-images-idx3-ubyte",
    "../data/MNIST/raw/train-labels-idx1-ubyte"
)


# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True    # 确保每个epoch训练数据随机打乱（Epoch = 把整个训练集完整训练一遍）
)


# 取出一个 batch
images, labels = next(iter(dataloader))


print("图片 batch 尺寸：", images.shape)
print("标签 batch 尺寸：", labels.shape)

print("第一张图片尺寸：", images[0].shape)
print("第一张图片标签：", labels[0])