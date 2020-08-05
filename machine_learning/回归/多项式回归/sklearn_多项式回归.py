"""
# @Time    :  2020/8/5
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # 生成多项式用的

# 读取数据
data = np.genfromtxt(r'job.csv', delimiter=',')
x_data = data[1:, 1]
y_data = data[1:, -1]

# plt.scatter(x_data, y_data)
# plt.show()

# 转换为二维数据
x_data = x_data[:, np.newaxis]  # 或者x_data = data[1:, 1, np.newaxis]
y_data = y_data[:, np.newaxis]  # 或者y_data = data[1:, -1, np.newaxis]

# 建模一元线性回归拟合
# model = LinearRegression()
# model.fit(x_data, y_data)

# 画线性回归线图看看效果
# plt.plot(x_data, y_data, 'b.')
# plt.plot(x_data, model.predict(x_data), 'r')
# plt.show()

# 定义多项式回归，degree的值可以调节多项式的特征
# degree = n，相当于n次方拟合
poly = PolynomialFeatures(degree=5)
# 特征处理
x_poly = poly.fit_transform(x_data)
# print(x_poly)
# 定义回归模型,并拟合
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_data)

# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(x_poly), 'r')  # predict 传的是x_poly,是处理后的数据
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''如果上面画出来10个点的线不够平滑，可以增加自变量，让他平滑一些'''

# 画图
plt.plot(x_data, y_data, 'b.')
x_test = np.linspace(1, 10, 50)  # 表示从1到10，之间有50个点
x_test = x_test[:, np.newaxis]
x_poly = poly.fit_transform(x_test)  # 一定要处理特征
plt.plot(x_test, lin_reg.predict(x_poly), 'r')  # predict 传的是x_poly,是处理后的数据
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
