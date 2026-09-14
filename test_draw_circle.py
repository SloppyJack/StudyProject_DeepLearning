import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 创建一个画布和坐标轴
fig, ax = plt.subplots()

# 创建一个圆形对象：参数分别为圆心坐标(0.5, 0.5)、半径0.2、边框颜色、填充颜色
circle = Circle((0.5, 0.5), radius=0.2, edgecolor='red', facecolor='none', linewidth=2)

# 将圆形添加到坐标轴中
ax.add_patch(circle)

# 设置坐标轴范围并保持长宽比一致（防止圆被压扁成椭圆）
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal', adjustable='box')

# 显示图形
plt.show()