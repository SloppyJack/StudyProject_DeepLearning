import torch
import torch.nn as nn

# 每个元素有 50% 的概率被置为 0
# 当p=0.5时，剩下的元素 y = x/(1-p)，即留下元素为原来两倍。
# dropout目的丢掉一半元素后，整体期望不变。0.5 * 0 + 0.5 * 2x = x
dropout = nn.Dropout(p=0.5) #

x = torch.tensor([
    1., 2., 3., 4.,
    5., 6., 7., 8.
])

print("原始 x：")
print(x)

dropout.train()

print("\ntrain 第一次：")
print(dropout(x))

print("\ntrain 第二次：")
print(dropout(x))

print("\ntrain 第三次：")
print(dropout(x))

dropout.eval()

print("\neval 第一次：")
print(dropout(x))

print("\neval 第二次：")
print(dropout(x))

print("\neval 第三次：")
print(dropout(x))