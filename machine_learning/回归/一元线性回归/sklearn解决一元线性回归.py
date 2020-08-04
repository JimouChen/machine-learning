"""
# @Time    :  2020/8/4
# @Author  :  Jimou Chen
"""
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]

plt.scatter(x_data, y_data)
plt.show()
print(x_data.shape)  # 打印出来是(100,)  ，表示一维向量，有100个数据

# 由于fit需要传入二维数据，所以需要对x_data和y_data做处理
# 给他们加个维度变成2维
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]
print(x_data.shape)  # 得到(100, 1)，为二维的

# 创建模型，并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)

plt.plot(x_data, y_data, 'b.')
# 预测y_data
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
