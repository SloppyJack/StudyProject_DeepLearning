import torch
from torch.utils.data import DataLoader

from mnist_dataset import MNISTDataset
from simple_cnn import SimpleCNN



# =========================
# 1. 准备测试集
# =========================

train_dataset = MNISTDataset(
    image_path="../data/MNIST/raw/train-images-idx3-ubyte",
    label_path="../data/MNIST/raw/train-labels-idx1-ubyte"
)
test_dataset = MNISTDataset(
    image_path="../data/MNIST/raw/t10k-images-idx3-ubyte",
    label_path="../data/MNIST/raw/t10k-labels-idx1-ubyte"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model = SimpleCNN()
# 加载模型参数
model.load_state_dict(
    torch.load("mnist_cnn.pth")
)
# 切换到测试模式
model.eval()

# 计算训练集准确率
correct = 0
total = 0

with torch.no_grad():

    for images, labels in train_dataloader:

        output = model(images)

        predictions = output.argmax(dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

accuracy = correct / total

print(f"训练集准确率：{accuracy * 100:.2f}%")

# 计算测试集准确率
correct = 0
total = 0

# 不计算梯度
with torch.no_grad():

    for images, labels in test_dataloader:

        output = model(images)

        predictions = output.argmax(dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

accuracy = correct / total

print(f"测试集准确率：{accuracy * 100:.2f}%")

print("测试集数量:", len(test_dataset))

images, labels = next(iter(test_dataloader))

print("图片 batch 尺寸:", images.shape)
print("标签 batch 尺寸:", labels.shape)