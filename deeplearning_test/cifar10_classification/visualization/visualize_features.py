
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from datasets.cifar10_dataset import get_test_dataloader
from models.simple_cnn import SimpleCNN


# 1. 设置路径
PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "outputs/checkpoints/best_model.pth"
SAVE_DIR = PROJECT_DIR / "outputs/visualization/feature_maps"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# 2. 创建并加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()


# 3. 获取一张测试图片
test_dataloader = get_test_dataloader()

images, labels = next(iter(test_dataloader))

# 取 Batch 中的第一张图片
image = images[0]

# 将标准化后的图片恢复为正常颜色
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

original_image = image.cpu() * std + mean
original_image = original_image.clamp(0, 1)

# 保存原始图片
save_image(original_image, SAVE_DIR / "original.png")

print(f"图片尺寸：{image.shape}")
print(f"真实标签：{labels[0].item()}")


# 4. 保存卷积特征图的字典
feature_maps = {}


# 5. 定义 Forward Hook
def get_hook(name):
    def hook(module, inputs, output):
        feature_maps[name] = output.detach().cpu()

    return hook


# 6. 注册三个卷积层的 Hook
handles = [
    model.conv1.register_forward_hook(get_hook("conv1")),
    model.conv2.register_forward_hook(get_hook("conv2")),
    model.conv3.register_forward_hook(get_hook("conv3"))
]


# 7. 前向传播
try:
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
finally:
    # 用完后移除 Hook
    for handle in handles:
        handle.remove()


# 8. 保存特征图
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


print("三层卷积特征图保存完成！")
