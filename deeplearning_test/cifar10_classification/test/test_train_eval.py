import torch

from datasets.cifar10_dataset import get_train_dataloader
from models.simple_cnn import SimpleCNN

# 加载数据
train_dataloader = get_train_dataloader()

# 创建模型并加载训练好的权重
model = SimpleCNN()

state_dict = torch.load(
    "../outputs/checkpoints/best_model.pth",
    weights_only=True,
    map_location="cpu"
)
model.load_state_dict(state_dict)

# 取一批图片
images, labels = next(iter(train_dataloader))

# train 模式
model.train()

with torch.no_grad():
    train_outputs = model(images)
    train_preds = train_outputs.argmax(dim=1)
    train_acc = (train_preds == labels).float().mean()

# eval 模式
model.eval()

with torch.no_grad():
    eval_outputs = model(images)
    eval_preds = eval_outputs.argmax(dim=1)
    eval_acc = (eval_preds == labels).float().mean()

print(f"train模式准确率: {train_acc.item() * 100:.2f}%")
print(f"eval模式准确率: {eval_acc.item() * 100:.2f}%")