import numpy as np

# 目的：使用RNN递归神经网络检测序列中是否存在两个1
# 重要的两个基本公式：
# 1. h_t=ReLU(w_hh @ h_t_prev + w_xh * x)，其中w_xh * x只会存在列向量(1, 0, 0) 或(0，0，0)，则h_t表示当前的隐藏
# 状态，h_t=(current, previous, 1)
# 2. y_t=ReLU(w_yh @ h_t)，只有在h_t=(1, 1, 1)时才输出y_t=1，其余情况均输出0



# 1. 定义 ReLU 激活函数
def relu(x):
    return np.maximum(0, x)


# 2. w_xh为当前输入的权重矩阵
w_xh = np.array([[1], [0], [0]])

# w_hh为上一次输入的权重矩阵
w_hh = np.array([[0, 0, 0], # current以当前输入x为准，此处全置零
                 [1, 0, 0],  # 负责将上一步的“Current”复制为“Previous”
                 [0, 0, 1]])  # 保留常量 1

# w_yh为输出结果的权重矩阵
w_yh = np.array([1, 1, -1])

x_seq = [0, 1, 0, 1, 1, 1, 0, 1, 1]

# 3. 初始化上一步的隐藏状态 h_t_prev
# 初始状态：Current=0, Previous=0, 常量=1，默认碰到了两个0
h_t_prev = np.array([[0], [0], [1]])

# 用于存储每个时间步的输出结果
y_seq = []

print("开始 RNN 前向传播计算：\n")
print(f"{'时间步 t':<8} | {'输入 x':<6} | {'当前隐藏状态 h_t':<20} | {'输出 y_t'}")
print("-" * 55)

# 4. 循环遍历序列
for t, x in enumerate(x_seq):
    # 计算当前隐藏状态 h_t，对应公式: h_t = ReLU(W_hh @ h_{t-1} + W_xh @ x)
    h_t = relu(w_hh @ h_t_prev + (w_xh * x))
    # 计算当前输出 y_t，对应公式: y_t = ReLU(W_hy @ h_t)
    y_t = relu(w_yh @ h_t)
    y_seq.append(y_t[0])

    # h_t.flatten() 是为了让打印出来像 [a, b, c] 而不是 [[a], [b], [c]]
    print(f"{t:<10} | {x:<8} | {str(h_t.flatten()):<20} | {y_t[0]}")

    # 5. 更新状态
    h_t_prev = h_t

print("-" * 55)
print(f"\n最终输入序列 X: {x_seq}")
print(f"最终输出序列 Y: {y_seq}")