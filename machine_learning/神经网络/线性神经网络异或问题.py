"""
# @Time    :  2020/8/8
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt

'''x0,x1,x2,x1^2,x1x2,x2^2'''

# 输入数据
X = np.array([[1, 0, 0, 0, 0, 0],
              [1, 1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0, 1],
              [1, 1, 1, 1, 1, 1]])
# 标签
Y = np.array([-1, 1, 1, -1])

# 初始化权值，有6个权值
W = (np.random.random(6) - 0.5) * 2
# 设置学习率
lr = 0.11


# 定义更新权值的函数
def update_weight():
    global X, Y, W, lr
    out = np.dot(X, W.T)  # 计算当前输出  # 神经网络输出,直接得到4个预测值
    theta_w = lr * ((Y - out.T).dot(X)) / int(X.shape[0])  # 数据量大时取平均
    W = W + theta_w


# 定义计算预测结果的函数
def calculate(x, root_num):
    global W
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + x * W[1] + x * x * W[3]
    if root_num == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if root_num == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


# 通过增加循环次数，使得分类效果越好
for i in range(1000):
    update_weight()

'''上面迭代到最后一次时，W就可以确定分界线的截距和效率了'''
'''可以把图画出来'''

# 正样本
x1 = [0, 1]
y1 = [1, 0]
# 负样本
x2 = [1, 0]
y2 = [1, 0]
# 画图横坐标边界
x_range = np.linspace(-1, 3)  # 或者x_range = [0, 5]

plt.figure()
plt.plot(x_range, calculate(x_range, 1), 'r')
plt.plot(x_range, calculate(x_range, 2), 'r')
plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'yo')
plt.show()
