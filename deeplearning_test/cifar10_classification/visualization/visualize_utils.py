from pathlib import Path
from torchvision.utils import save_image

def save_tensor_image(image, save_path, mean=None, std=None):
    """保存 Tensor 图片，可选反归一化。"""

    image = image.detach().cpu().clone()

    if mean is not None and std is not None:
        import torch
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        image = image * std + mean

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(image.clamp(0, 1), save_path)