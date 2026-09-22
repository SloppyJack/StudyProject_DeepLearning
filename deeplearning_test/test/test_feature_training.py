import torch
import matplotlib.pyplot as plt

from deeplearning_test.simple_cnn import SimpleCNN
from deeplearning_test.mnist_data import get_test_dataloader

def save_image(arr, name):
    plt.imsave(
        f"../train_imgs/{name}.png",
        arr,
        cmap="gray"
    )

# 取测试集中的一个 batch
test_dataloader = get_test_dataloader()

images, labels = next(iter(test_dataloader))

# 只取第一张图片
image = images[0:1] # 切片images下标0~1 -> 1x1x28x28
label = labels[0]

# 保存原始图片
plt.imsave(
    "../train_imgs/original_image.png",
    image[0, 0].numpy(),
    cmap="gray"
)

# 创建一个 CNN
model = SimpleCNN()

# 加载 Epoch 0 的参数
model.load_state_dict(
    torch.load("../pth/model_epoch_0.pth")
)

kernel_epoch_0 = model.conv1.weight[0, 0]
save_image(kernel_epoch_0.detach().numpy(), "epoch_0_conv1_kernel0")
print("kernel_epoch_0: ", kernel_epoch_0)

model.eval()

with torch.no_grad():
    # ① 第一层卷积
    conv1_output = model.conv1(image)
    save_image(conv1_output[0, 0].numpy(), "epoch_0_conv1_map0")

    # ② ReLU
    relu1_output = torch.relu(conv1_output)
    save_image(relu1_output[0, 0].numpy(), "epoch_0_relu1_map0")

    # ③ MaxPool
    pool1_output = model.pool(relu1_output)
    save_image(pool1_output[0, 0].numpy(), "epoch_0_poo1_map0")

    print("Conv1:", conv1_output[0, 0].min().item(),
          conv1_output[0, 0].max().item())

    print("ReLU1:", relu1_output[0, 0].min().item(),
          relu1_output[0, 0].max().item())

    print("Pool1:", pool1_output[0, 0].min().item(),
          pool1_output[0, 0].max().item())

    for channel in range(16):
        feature_map = conv1_output[0, channel]

        save_image(
            feature_map.numpy(),
            f"epoch_0_conv1_map{channel}"
        )

# ==========================
# 加载 Epoch 5
# ==========================

model.load_state_dict(
    torch.load("../pth/model_epoch_5.pth")
)

kernel_epoch_5 = model.conv1.weight[0, 0]
save_image(kernel_epoch_5.detach().numpy(), "epoch_5_conv1_kernel0")
print("kernel_epoch_5: ", kernel_epoch_5)

model.eval()

with torch.no_grad():

    # Epoch 5 的第一层卷积
    conv1_output_epoch5 = model.conv1(image)

    # 保存 16 个通道
    for channel in range(16):

        feature_map = conv1_output_epoch5[0, channel]

        save_image(
            feature_map.numpy(),
            f"epoch_5_conv1_map{channel}"
        )