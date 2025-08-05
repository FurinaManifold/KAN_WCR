import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde

# 假设你已经有了一组二维向量，这里我们随机生成一些数据作为示例
np.random.seed(0)
x = np.random.normal(size=1000)
y = x * 3 + np.random.normal(size=1000)

# 创建网格
nbins = 300
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]

# 计算密度
positions = np.vstack([xi.ravel(), yi.ravel()])
values = np.vstack([x, y])
kernel = kde(values)
zi = np.reshape(kernel(positions).T, xi.shape)

# 绘制热力图
fig, ax = plt.subplots()
ax.imshow(np.rot90(zi), cmap=plt.cm.Blues, extent=[x.min(), x.max(), y.min(), y.max()])
# 或者使用等高线填充
# cs = ax.contourf(xi, yi, zi, cmap=plt.cm.Blues)

# 设置图表属性
ax.plot(x, y, 'k.', markersize=2)  # 可选：显示原始数据点
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([y.min(), y.max()])
plt.savefig("1.png")