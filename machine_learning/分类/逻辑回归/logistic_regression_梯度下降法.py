"""
# @Time    :  2020/8/6
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing

# False的话不做数据标准化
scale = False


# 逻辑回归预测函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 代价函数,传入训练数据，标签，权值，三者都是矩阵形式
def cost(x_mat, y_mat, ws):
    left = np.multiply(y_mat, np.log(sigmoid(x_mat * ws)))
    right = np.multiply(1 - y_mat, np.log(1 - sigmoid(x_mat * ws)))
    return np.sum(left + right) / -(len(x_mat))


# 求梯度，用梯度改变权值
def gradAscent(xArr, yArr):
    if scale:
        xArr = preprocessing.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.001
    epochs = 10000
    costList = []
    # 计算数据行列数
    # 行代表数据个数，列代表权值个数
    m, n = np.shape(xMat)
    # 初始化权值
    ws = np.mat(np.ones((n, 1)))

    for i in range(epochs + 1):
        # xMat和weights矩阵相乘
        h = sigmoid(xMat * ws)
        # 计算误差
        ws_grad = xMat.T * (h - yMat) / m
        ws = ws - lr * ws_grad

        if i % 50 == 0:
            costList.append(cost(xMat, yMat, ws))
    return ws, costList


data = np.genfromtxt('LR-testSet.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]


# 切分数据,并画出散点图
def split_plot():
    # 切分成两种类别，0和1
    # x0和y0是0类别的那两列数据，x1，y1同理
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])

    # 画出散点图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    # 画图例
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')


# 切分数据,并画出散点图
split_plot()
plt.show()
# 把y_data变成二维
y_data = y_data[:, np.newaxis]
# 给样本提价偏置项
X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)
# 训练模型，得到权值和cost值的变化
ws, costList = gradAscent(X_data, y_data)
print(ws)

if scale == False:
    # 画图决策边界
    split_plot()
    x_test = [[-4], [3]]
    y_test = (-ws[0] - x_test * ws[1]) / ws[2]
    plt.plot(x_test, y_test, 'k')
    plt.show()

# 画图 loss值的变化
x = np.linspace(0, 10000, 201)
plt.plot(x, costList, c='r')
plt.title('Train')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()


# 预测
def predict(x_data, ws):
    if scale == True:
        x_data = preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat * ws)]


predictions = predict(X_data, ws)

print(classification_report(y_data, predictions))


