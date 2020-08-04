"""
# @Time    :  2020/8/4
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 读入数据
data = np.genfromtxt(r'Delivery.csv', delimiter=',')
# 切分数据
x_data = data[:, :-1]
y_data = data[:, -1]

# 创建模型并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)
print(model)

# 打印系数,有几个自变量打印出来就有几个系数
print('系数：', model.coef_)
# 打印截距
print('截距：', model.intercept_)

# 测试
x_test = [[102, 4]]
predict = model.predict(x_test)
print('预测值：', predict)

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)  # 点为红色三角形
x0 = x_data[:, 0]
x1 = x_data[:, 1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]
# 画3D图
ax.plot_surface(x0, x1, z)
# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# 显示图像
plt.show()
