
from pathlib import Path
from datasets.cifar10_classes import num_classes
import torch
import os


# ================= 项目路径 =================

# 当前文件所在目录，也就是 cifar10_classification
BASE_DIR = Path(__file__).resolve().parent

# 数据集路径
DATA_DIR = BASE_DIR / "data"

# 输出路径
OUTPUT_DIR = BASE_DIR / "outputs"

# 模型权重保存路径
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# 可视化图片保存路径
FIGURE_DIR = OUTPUT_DIR / "figures"

# CIFAR10数据源均值和方差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ================= 模型配置 =================

MODEL_NAME = "resnet18"

NUM_CLASSES = num_classes


# ================= 训练配置 =================

BATCH_SIZE = 128

EPOCHS = 15

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0

SCHEDULER_T_MAX = 15
SCHEDULER_ETA_MIN = 1e-6


# ================= 学习率调度 =================

STEP_SIZE = 5

GAMMA = 0.5


# ================= Early Stopping =================

PATIENCE = 5


# ================= 设备配置 =================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# CPU 计算线程数
CPU_NUM_THREADS = 16

if DEVICE.type == "cpu":
    torch.set_num_threads(CPU_NUM_THREADS)

# ================= 自动创建输出目录 =================

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# 系统配置相关

print("CPU逻辑线程数：", os.cpu_count())
print("PyTorch默认线程数：", torch.get_num_threads())
print("PyTorch线程池：", torch.get_num_interop_threads())
