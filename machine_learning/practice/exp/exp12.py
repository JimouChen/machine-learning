"""
# @Time    :  2020/10/15
# @Author  :  Jimou Chen
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


# 定义一个产生二维数据的函数，通过改变方差和均值来改变位置
def product_num(num, theta_x, mean_x, theta_y, mean_y):
    x = np.random.randn(num, 1) * theta_x + mean_x
    y = np.random.randn(num, 1) * theta_y + mean_y
    data = np.concatenate((x, y), axis=1)

    return x, y, data


if __name__ == '__main__':
    # 生成数据特征
    x1, y1, data1 = product_num(400, 1, 0, 1, 0)
    x2, y2, data2 = product_num(400, 1, 1, 1.5, 4)
    x3, y3, data3 = product_num(400, 2, 8, 1, 5)
    x4, y4, data4 = product_num(400, 0.5, 3, 0.5, 3)

    # 设置标签，4个类
    label1 = np.zeros(400, dtype=int)
    label2 = np.zeros(400, dtype=int) + 1
    label3 = np.zeros(400, dtype=int) + 2
    label4 = np.zeros(400, dtype=int) + 3

    # 画图
    plt.scatter(x1, y1, c='b', marker='^', label='class1')
    plt.scatter(x2, y2, c='r', marker='v', label='class2')
    plt.scatter(x3, y3, c='y', marker='<', label='class3')
    plt.scatter(x4, y4, c='g', marker='>', label='class4')
    plt.legend()
    plt.show()

    # 合并特征和标签
    data = np.concatenate((data1, data2, data3, data4), axis=0)
    label = np.concatenate((label1, label2, label3, label4), axis=0)

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, label)

    # 建模拟合
    model = LinearRegression()
    # model = LogisticRegression()
    model.fit(x_train, y_train)

    # 进行预测类别
    pred1 = model.predict(x_test)
    prediction = model.predict(data)
    # 由于调用了线性回归分类器，所以得到的结果是连续的，所以转换成离散的整数
    prediction = np.trunc(np.array(prediction)).astype(int)

    # 显示数据
    np.set_printoptions(threshold=10000)
    print(prediction)

    # 准确率
    print(model.score(x_test, y_test))
