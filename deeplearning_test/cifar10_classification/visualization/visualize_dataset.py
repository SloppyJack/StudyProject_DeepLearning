from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


dataset = CIFAR10(
    root="../data",
    train=True,
    download=False,
    transform=transform
)


for i in range(10):

    # 注意：始终读取第 0 张
    image, label = dataset[0]

    save_image(
        image,
        f"../outputs/flip_test_{i}.png"
    )

    print(
        f"第 {i + 1} 次读取完成，"
        f"label={label}"
    )