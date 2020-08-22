"""
# @Time    :  2020/8/22
# @Author  :  Jimou Chen
"""
import matplotlib.pyplot as plt
from sklearn import datasets

'''用于解决非线性问题'''

# 制造数据
x_data, y_data = datasets.make_circles(n_samples=500, factor=0.3, noise=0.1)
# 画出来看看
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

'''接下来把2维映射到3维'''

z_data = x_data[:, 0] ** 2 + x_data[:, 1] ** 2
# 画3d图
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], z_data, c=y_data, s=10)  # s是大小
plt.show()