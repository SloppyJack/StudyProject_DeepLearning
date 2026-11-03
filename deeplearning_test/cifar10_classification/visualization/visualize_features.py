

import torch
from torchvision.utils import save_image, make_grid

from datasets.cifar10_dataset import get_test_dataloader
from experiment_config import MODEL_NAME, DEVICE, CHECKPOINT_DIR, OUTPUT_DIR, NUM_CLASSES, CIFAR10_STD, CIFAR10_MEAN
from models.resnet18 import ResNet18
from models.simple_cnn import SimpleCNN


# ================= 1. 模型配置 =================
MODEL_CONFIG = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18
}
if MODEL_NAME not in MODEL_CONFIG:
    raise ValueError(f"不支持的模型名称：{MODEL_NAME}")

# ================= 2. 设置路径 =================
MODEL_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"

# 不同模型分别保存，避免覆盖
SAVE_DIR = OUTPUT_DIR / "visualization" / "feature_maps" / MODEL_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ================= 3. 创建并加载模型 =================
model = MODEL_CONFIG[MODEL_NAME](
    num_classes=NUM_CLASSES
).to(DEVICE)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

print(f"当前可视化模型：{MODEL_NAME}")
print(f"权重路径：{MODEL_PATH}")


# ================= 4. 获取一张测试图片 =================
test_dataloader = get_test_dataloader()

images, labels = next(iter(test_dataloader))

# 取 Batch 中的第一张图片
image = images[0]

# 将标准化后的图片恢复为正常颜色
mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

original_image = image.cpu() * std + mean
original_image = original_image.clamp(0, 1)

# 保存原始图片
save_image(original_image, SAVE_DIR / "original.png")

print(f"图片尺寸：{image.shape}")
print(f"真实标签：{labels[0].item()}")


# 5. 保存卷积特征图的字典
feature_maps = {}


# 6. 定义 Forward Hook
def get_hook(name):
    def hook(module, inputs, output):
        feature_maps[name] = output.detach().cpu()  # 分离tensor并放到cpu中

    return hook


# 7. 根据模型注册Hook
if MODEL_NAME == "simple_cnn":

    handles = [
        model.conv1.register_forward_hook(get_hook("conv1")),
        model.conv2.register_forward_hook(get_hook("conv2")),
        model.conv3.register_forward_hook(get_hook("conv3"))
    ]

elif MODEL_NAME == "resnet18":

    handles = [
        model.conv1.register_forward_hook(get_hook("conv1")),
        model.layer1[0].conv1.register_forward_hook(
            get_hook("layer1")
        ),
        model.layer2[0].conv1.register_forward_hook(
            get_hook("layer2")
        ),
        model.layer3[0].conv1.register_forward_hook(
            get_hook("layer3")
        )
    ]



# 8. 前向传播
try:
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(DEVICE))
finally:
    # 用完后移除 Hook
    for handle in handles:
        handle.remove()


# 9. 保存特征图
for name, features in feature_maps.items():

    print(f"{name} 原始输出尺寸：{features.shape}")

    # [1, C, H, W] -> [C, 1, H, W]
    features = features[0].unsqueeze(1)

    # 对每个通道分别归一化
    min_vals = features.amin(dim=(2, 3), keepdim=True)
    max_vals = features.amax(dim=(2, 3), keepdim=True)

    features = (
        (features - min_vals)
        / (max_vals - min_vals + 1e-8)
    )

    # 将所有通道拼接为网格
    grid = make_grid(
        features,
        nrow=8,
        padding=2
    )

    save_image(grid, SAVE_DIR / f"{name}.png")

    # 额外保存前 4 个通道的特征图
    selected_features = features[:4]

    selected_grid = make_grid(
        selected_features,
        nrow=2,
        padding=4,
        pad_value=1
    )

    save_image(
        selected_grid,
        SAVE_DIR / f"{name}_selected.png"
    )

    print(f"{name} 已保存")


print(f"{MODEL_NAME} 特征图保存完成！")
print(f"保存目录：{SAVE_DIR}")