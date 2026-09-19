import torch
from torch.utils.data import Dataset
import numpy as np


class MNISTDataset(Dataset):

    def __init__(self, image_path, label_path):
        with open(image_path,'rb') as f:    # 以二进制只读方式打开文件，用完自动关闭
            # np.frombuffer将字节流转换成数组
            images = np.frombuffer(
                f.read(),   # 一次性读完
                np.uint8,   # 数据类型：无符号8位整数，0~255
                offset=16   # 跳过前16Byte
            )
        # 将一维数组转为三维数组6000x28x28
        self.images = images.reshape(-1,28,28) # 第一个参数-1为NumPy的智能占位符

        with open(label_path,'rb') as f:
            labels = np.frombuffer(
                f.read(),
                np.uint8,
                offset=8
            )
        self.labels = labels


    # 数据集大小
    def __len__(self):
        return len(self.labels)


    # 按编号取出图片和对应标签
    def __getitem__(self,index):
        image = self.images[index]
        label = self.labels[index]

        # 转成二维PyTorch张量
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # 归一化
        image = image / 255.0


        # CNN输入需要：
        # [channel,height,width]
        # 给图片增加通道，实际变成了三维张量
        image = image.unsqueeze(0)


        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return image,label