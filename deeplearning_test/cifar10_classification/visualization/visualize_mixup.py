
import torch

from datasets.cifar10_dataset import get_train_dataloader
from experiment_config import OUTPUT_DIR, CIFAR10_MEAN, CIFAR10_STD
from visualize_utils import save_tensor_image


def visualize_mixup():
    train_dataloader = get_train_dataloader()

    images, labels = next(iter(train_dataloader))

    # 取同一 Batch 中的前两张图片
    image1 = images[0]
    image2 = images[1]

    label1 = labels[0].item()
    label2 = labels[1].item()

    save_dir = OUTPUT_DIR / "mixup"

    # 保存两张原始图片
    save_tensor_image(
        image1,
        save_dir / "image1.png",
        CIFAR10_MEAN,
        CIFAR10_STD
    )

    save_tensor_image(
        image2,
        save_dir / "image2.png",
        CIFAR10_MEAN,
        CIFAR10_STD
    )

    # 使用不同的混合比例
    for lam in [0.2, 0.5, 0.8]:
        mixed_image = lam * image1 + (1 - lam) * image2

        save_tensor_image(
            mixed_image,
            save_dir / f"mixup_{lam}.png",
            CIFAR10_MEAN,
            CIFAR10_STD
        )

    print(f"图片1类别：{label1}")
    print(f"图片2类别：{label2}")
    print(f"图片已保存到：{save_dir}")


if __name__ == "__main__":
    visualize_mixup()
