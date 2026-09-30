import torch

from models.simple_cnn import SimpleCNN
from datasets.cifar10_dataset import get_train_dataloader


# 创建一个全新的模型
model = SimpleCNN()

# 获取训练数据
train_dataloader = get_train_dataloader()


# =========================
# 1. 刚初始化
# =========================

print("========== 初始化 ==========")

print("running_mean:")
print(model.bn1.running_mean)

print("running_var:")
print(model.bn1.running_var)


# =========================
# 2. train 模式
# =========================

model.train()

print("\n========== train模式 ==========")

# 只跑5个batch，不训练
for batch_index, (images, labels) in enumerate(train_dataloader):

    # 只进行前向传播
    with torch.no_grad():
        outputs = model(images)

    if batch_index == 4:
        break


print("运行5个batch后 running_mean:")
print(model.bn1.running_mean)

print("运行5个batch后 running_var:")
print(model.bn1.running_var)


# =========================
# 3. eval 模式
# =========================

model.eval()

print("\n========== eval模式 ==========")

# 先保存一份，方便后面比较
mean_before = model.bn1.running_mean.clone()
var_before = model.bn1.running_var.clone()


# 再跑5个batch
for batch_index, (images, labels) in enumerate(train_dataloader):

    with torch.no_grad():
        outputs = model(images)

    if batch_index == 4:
        break


print("eval前 running_mean:")
print(mean_before)

print("eval后 running_mean:")
print(model.bn1.running_mean)

print()

print("eval前 running_var:")
print(var_before)

print("eval后 running_var:")
print(model.bn1.running_var)

print(
    "running_mean 是否发生变化：",
    not torch.equal(mean_before, model.bn1.running_mean)
)

print(
    "running_var 是否发生变化：",
    not torch.equal(var_before, model.bn1.running_var)
)

print("====== model.parameters() ======")

for name, parameter in model.named_parameters():
    print(name)

print("====== model.state_dict() ======")

for name in model.state_dict():
    print(name)

print("====== model.named_buffers() ======")

for name, buffer in model.named_buffers():
    print(name, buffer)