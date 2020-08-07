"""
# @Time    :  2020/8/7
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt

'''激活函数是y = x, 效果比单层感知器好'''

# 输入数据
X = np.array([[1, 3, 3],
              [1, 4, 3],
              [1, 1, 1],
              [1, 0, 2]])
# 标签
Y = np.array([[1],
              [1],
              [-1],
              [-1]])

# 初始化权值，3行1列，取值-1到1
W = (np.random.random([3, 1]) - 0.5) * 2
# 设置学习率
lr = 0.11


# 定义更新权值的函数
def update_weight():
    global X, Y, W, lr
    out = np.dot(X, W)  # 计算当前输出  # 神经网络输出,直接得到4个预测值
    theta_w = lr * (X.T.dot(Y - out)) / int(X.shape[0])  # 数据量大时取平均
    W = W + theta_w


# 通过增加循环次数，使得分类效果越好
for i in range(1000):
    update_weight()
    # print('第{}次迭代：'.format(i))
    # print(W)
    # out = np.dot(X, W)  # 计算当前输出
    # 其实这里是没有用的因为实际要达到和预测一样相差太远了，所以可以直接通过迭代次数来规定收敛
    # .all()只有输出的所有预测值都与实际输出一样，才说明模型收敛，循环结束
    # if (out == Y).all():
    #     print('Finished , epoch:', i)
    #     break


'''上面迭代到最后一次时，W就可以确定分界线的截距和效率了'''

# 计算分界线的斜率和截距
k = -W[1] / W[2]
b = -W[0] / W[2]
print('k = ', k)
print('b = ', b)

'''可以把图画出来'''

# 正样本
x1 = [3, 4]
y1 = [3, 3]
# 负样本
x2 = [1, 0]
y2 = [1, 2]
# 画图横坐标边界
x_range = (0, 5)  # 或者x_range = [0, 5]

plt.figure()
plt.plot(x_range, x_range * k + b, 'r')
plt.scatter(x1, y1, c='b')
plt.scatter(x2, y2, c='y')
plt.show()
