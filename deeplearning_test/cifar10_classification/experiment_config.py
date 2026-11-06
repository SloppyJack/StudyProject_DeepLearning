
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

# 预训练主干使用 ImageNet 归一化；变量名保留以兼容可视化。
CIFAR10_MEAN = (0.485, 0.456, 0.406)
CIFAR10_STD = (0.229, 0.224, 0.225)


# ================= 模型配置 =================

MODEL_NAME = "resnet18"

NUM_CLASSES = num_classes

# 96像素是CPU预算下的输入适配，不会为32像素原图增加细节。
INPUT_SIZE = 96
PRETRAINED_WEIGHTS = BASE_DIR / "outputs/pretrained/resnet18-f37072fd.pth"
BACKBONE_LEARNING_RATE = 0.0001


# ================= 训练配置 =================

BATCH_SIZE = 128

EPOCHS = 15

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

USE_MIXUP = True
MIXUP_ALPHA = 0.1
MIXUP_EPOCHS = 10


SCHEDULER_T_MAX = 15
SCHEDULER_ETA_MIN = 1e-6
ONECYCLE_PCT_START = 0.3
ONECYCLE_DIV_FACTOR = 10.0
ONECYCLE_FINAL_DIV_FACTOR = 100.0


# ================= 学习率调度 =================

STEP_SIZE = 5

GAMMA = 0.5


# ================= Early Stopping =================

# 本轮固定完整15轮，不提前结束迁移学习对照。
PATIENCE = EPOCHS


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
