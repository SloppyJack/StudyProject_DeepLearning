import torch

from datasets.cifar10_dataset import get_test_dataloader
from models.simple_cnn import SimpleCNN


# 获取测试数据
test_dataloader = get_test_dataloader()


# 创建模型
model = SimpleCNN()


# 加载训练好的参数
state_dict = torch.load("../outputs/checkpoints/best_model.pth", weights_only=True)
model.load_state_dict(state_dict)

print(state_dict.keys())


# 切换到测试模式
model.eval()


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_dataloader:

        # 前向传播
        outputs = model(images)

        # 获取预测类别
        predicted = outputs.argmax(dim=1)

        # 样本总数
        total += labels.size(0)

        # 预测正确数量
        correct += (predicted == labels).sum().item()


accuracy = correct / total

print("测试图片数量：", total)
print("预测正确数量：", correct)
print(f"测试准确率：{accuracy * 100:.2f}%")