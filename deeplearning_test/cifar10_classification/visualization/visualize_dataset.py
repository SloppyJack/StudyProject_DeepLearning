
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image


# 保存目录
SAVE_DIR = Path("../outputs/visualization/augmentation")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 加载原始数据，不执行任何增强
dataset = CIFAR10(
    root="../data",
    train=True,
    download=False,
    transform=None
)

# 始终使用同一张原始图片
original_image, label = dataset[0]

to_tensor = transforms.ToTensor()

# 保存原图
save_image(
    to_tensor(original_image),
    SAVE_DIR / "original.png"
)

# 1. 仅随机裁剪
random_crop = transforms.RandomCrop(32, padding=4)

save_image(
    to_tensor(random_crop(original_image)),
    SAVE_DIR / "random_crop.png"
)

# 2. 仅水平翻转
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

save_image(
    to_tensor(horizontal_flip(original_image)),
    SAVE_DIR / "horizontal_flip.png"
)

# 3. 组合增强（与训练阶段一致）
augmentation = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

for i in range(10):
    image = augmentation(original_image)

    save_image(
        image,
        SAVE_DIR / f"augmented_{i + 1}.png"
    )

    print(f"第 {i + 1} 次增强完成，label={label}")
