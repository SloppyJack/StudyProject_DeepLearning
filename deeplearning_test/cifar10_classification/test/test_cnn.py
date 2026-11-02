import torch
from torch import nn

from datasets.cifar10_classes import CIFAR10_CLASSES
from datasets.cifar10_dataset import get_test_dataloader
from experiment_config import MODEL_NAME, DEVICE, CHECKPOINT_DIR, NUM_CLASSES
from models.resnet18 import ResNet18
from models.simple_cnn import SimpleCNN


# 获取测试数据
test_dataloader = get_test_dataloader()
criterion = nn.CrossEntropyLoss()

# 模型配置
MODEL_CONFIG = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18
}
if MODEL_NAME not in MODEL_CONFIG:
    raise ValueError(f"不支持的模型名称: {MODEL_NAME}")

# 创建模型
model = MODEL_CONFIG[MODEL_NAME](num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# 加载训练好的参数
BEST_MODEL_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE,weights_only=True)
model.load_state_dict(state_dict)

print(f"当前测试模型：{MODEL_NAME}")
print(f"加载权重：{BEST_MODEL_PATH}")
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# 切换到测试模式
model.eval()

test_loss_sum = 0.0
test_correct = 0
test_total = 0

class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        output = model(images)
        loss = criterion(output, labels)

        test_loss_sum += loss.item() * labels.size(0)

        predicted = output.argmax(dim=1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

for class_id, class_name in CIFAR10_CLASSES.items():
    accuracy = class_correct[class_id] / class_total[class_id]

    print(f"{class_name}: {accuracy * 100:.2f}%")
test_loss = test_loss_sum / test_total
test_accuracy = test_correct / test_total

print(f"最终 Test Loss: {test_loss:.4f}")
print(f"最终 Test Accuracy: {test_accuracy:.2%}")