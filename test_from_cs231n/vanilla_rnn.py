import numpy as np


# 1. 定义 ReLU 激活函数
def relu(x):
    return np.maximum(0, x)


# 2. 初始化权重矩阵和输入序列
w_xh = np.array([[1], [0], [0]])

w_hh = np.array([[0, 0, 0],
                 [1, 0, 0],  # 红框所在行：负责将上一步的“Current”复制为“Previous”
                 [0, 0, 1]])  # 保留常量 1

w_yh = np.array([1, 1, -1])

x_seq = [0, 1, 0, 1, 1, 1, 0, 1, 1]

# 3. 初始化上一步的隐藏状态 h_t_prev
# 初始状态：Current=0, Previous=0, 常量=1
h_t_prev = np.array([[0], [0], [1]])

# 用于存储每个时间步的输出结果
y_seq = []

print("开始 RNN 前向传播计算：\n")
print(f"{'时间步 t':<8} | {'输入 x':<6} | {'当前隐藏状态 h_t':<20} | {'输出 y_t'}")
print("-" * 55)

# 4. 循环遍历序列
for t, x in enumerate(x_seq):
    # 计算当前隐藏状态 h_t
    # 对应公式: h_t = ReLU(W_hh @ h_{t-1} + W_xh @ x)
    # 注意：w_xh @ x 中，x 是标量，numpy 会自动进行广播（矩阵每个元素乘 x）
    h_t = relu(w_hh @ h_t_prev + (w_xh @ x))

    # 计算当前输出 y_t
    # 对应公式: y_t = ReLU(W_hy @ h_t)
    y_t = relu(w_yh @ h_t)

    # 将结果保存到列表（y_t 是一个一维数组，取第一个元素即可）
    y_seq.append(y_t[0])

    # 打印当前时间步的计算结果
    # h_t.flatten() 是为了让打印出来像 [a, b, c] 而不是 [[a], [b], [c]]
    print(f"{t:<10} | {x:<8} | {str(h_t.flatten()):<20} | {y_t[0]}")

    # 5. 更新状态
    # 对应代码注释：Copy over "current" value from previous hidden state to be "previous"
    h_t_prev = h_t

print("-" * 55)
print(f"\n最终输入序列 X: {x_seq}")
print(f"最终输出序列 Y: {y_seq}")