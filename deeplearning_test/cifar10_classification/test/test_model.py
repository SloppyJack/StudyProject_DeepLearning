from datasets.cifar10_dataset import get_train_dataloader
from models.simple_cnn import SimpleCNN


# 1. 获取 DataLoader
train_dataloader = get_train_dataloader()

# 2. 取一个 batch
images, labels = next(iter(train_dataloader))

print("输入 images shape:", images.shape)
print("labels shape:", labels.shape)


# 3. 创建 CNN 模型
model = SimpleCNN()

for key, value in model.state_dict().items():
    print(key, value.shape)

print("====== model.parameters() ======")

for name, parameter in model.named_parameters():
    print(name)


print("model:")
print(model)
print('weight.shape: ', model.conv3.weight.shape)
print('bias.shape: ', model.conv3.bias.shape)


# 4. 前向传播
outputs = model(images)

print("输出 outputs shape:", outputs.shape)

print("第1张图片的输出：")
print(outputs[0])

print("第1张图片真实标签：")
print(labels[0])