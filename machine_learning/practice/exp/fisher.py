"""
# @Time    :  2020/10/17
# @Author  :  Jimou Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# 定义一个产生二维数据的函数，通过改变方差和均值来改变位置
def product_num(num, theta_x, mean_x, theta_y, mean_y):
    x = np.random.randn(num, 1) * theta_x + mean_x
    y = np.random.randn(num, 1) * theta_y + mean_y
    data = np.concatenate((x, y), axis=1)

    return x, y, data


if __name__ == '__main__':
    x1, y1, data1 = product_num(200, 0.6, 1, 1.2, 3)
    x2, y2, data2 = product_num(200, 1.5, 4, 1.2, 0.5)
    label1 = np.zeros(200, dtype=int)
    label2 = np.ones(200, dtype=int)

    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2, c='b')
    plt.show()

    data = np.concatenate((data1, data2), axis=0)
    label = np.concatenate((label1, label2), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(data, label)
    LDA_clf = LinearDiscriminantAnalysis()
    LDA_clf.fit(x_train, y_train)

    print(LDA_clf.score(x_test, y_test))

