import torch
from torch import nn

from datasets.cifar10_dataset import get_test_dataloader
from models.simple_cnn import SimpleCNN


# 获取测试数据
test_dataloader = get_test_dataloader()
criterion = nn.CrossEntropyLoss()

# 创建模型
model = SimpleCNN()


# 加载训练好的参数
state_dict = torch.load("../outputs/checkpoints/best_model.pth", weights_only=True)
model.load_state_dict(state_dict)


# 切换到测试模式
model.eval()

test_loss_sum = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        output = model(images)
        loss = criterion(output, labels)

        test_loss_sum += loss.item() * labels.size(0)

        predicted = output.argmax(dim=1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_loss = test_loss_sum / test_total
test_accuracy = test_correct / test_total

print(f"最终 Test Loss: {test_loss:.4f}")
print(f"最终 Test Accuracy: {test_accuracy:.2%}")