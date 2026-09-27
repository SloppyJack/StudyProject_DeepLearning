import torch

from datasets.cifar10_dataset import get_train_dataloader

train_dataloader = get_train_dataloader()

#r_sum = g_sum = b_sum = 0
channel_sum = torch.zeros(3)
pixel_count = 0
for batch_idx, (images, labels) in enumerate(train_dataloader):
    # r_sum += images[:, 0, :, :].mean().item()
    # g_sum += images[:, 1, :, :].mean().item()
    # b_sum += images[:, 2, :, :].mean().item()
    channel_sum += images.sum(dim=(0, 2, 3)) # 利用张量特性，快速得到R G B像素值总和
    pixel_count += images.size(0) * images.size(2) * images.size(3)

# r_mean = r_sum / pixel_count
# g_mean = g_sum / pixel_count
# b_mean = b_sum / pixel_count
mean = channel_sum / pixel_count

squared_diff_sum = torch.zeros(3)
for batch_idx, (images, labels) in enumerate(train_dataloader):
    # tensor [64x3x32x32] - [1x3x1x1]，会自动broadcasting
    diff = images - mean.view(1, 3, 1, 1)   # mean.view: 使得[3] -> [1x3x1x1]
    # 张量中“*”是指对应元素相乘，不改变shape
    squared_diff = diff * diff
    squared_diff_sum += squared_diff.sum(dim=(0, 2, 3))

variance = squared_diff_sum / pixel_count
std = torch.sqrt(variance)

print(f"mean: {mean}")
print(f"std: {std}")