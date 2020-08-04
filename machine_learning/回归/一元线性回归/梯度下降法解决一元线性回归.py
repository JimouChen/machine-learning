"""
# @Time    :  2020/8/4
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt


# 最小二乘法，返回代价函数的值
def compute_error(b, k, x_data, y_data):
    total_error = 0
    for i in range(0, len(x_data)):
        total_error += (k * x_data[i] + b - y_data[i]) ** 2

    return total_error / (2.0 * len(x_data))  # 这里除以2可有可无


# 更新b和k
def get_bk(x_data, y_data, b, k, lr, epochs):
    # 计算总数据量
    m = len(x_data)
    # 循环epochs次
    for i in range(epochs):
        # 临时变量
        b_grad = 0
        k_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, m):
            b_grad += (1 / m) * (k * x_data[j] + b - y_data[j])
            k_grad += (1 / m) * (k * x_data[j] + b - y_data[j]) * x_data[j]
        # 更新b，k
        b = b - lr * b_grad
        k = k - lr * k_grad

    return b, k


# 载入数据
data = np.genfromtxt('data.csv', delimiter=',')
x_data = data[:, 0]  # 所有行都要，但只要第0列
y_data = data[:, 1]  # 只要第1列

# 画出分布图
plt.scatter(x_data, y_data)
plt.show()

'''接下来求解回归直线，先求那两个参数'''

# 先定义一些参数
# 设置学习率learning rate,截距，斜率
lr = 0.0001
b = 0
k = 0
# 最大迭代次数
epochs = 50

print('开始时：b = {}, k = {}, error = {}'.format(b, k, compute_error(b, k, x_data, y_data)))
print('正在建模......')
b, k = get_bk(x_data, y_data, b, k, lr, epochs)
print('迭代{}次后， b = {}, k = {}, error = {}'.format(epochs, b, k, compute_error(b, k, x_data, y_data)))

# 画出图像
plt.plot(x_data, y_data, 'b.')
# 画出回归直线,在同一张图上画
plt.plot(x_data, k * x_data + b, 'r')
plt.show()