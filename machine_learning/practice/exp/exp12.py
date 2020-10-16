"""
# @Time    :  2020/10/15
# @Author  :  Jimou Chen
"""
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def product_num(row, col, theta, mean):
    data = np.random.randn(row, col) * theta + mean
    x_data = data[:, 0]
    y_data = data[:, 1]
    return x_data, y_data, data

#
# def handle_pred_res(pred):
#
#     for i in range(len(pred)):
#         pred[i] = int(round(pred[i]))
#
#     return pred


if __name__ == '__main__':
    x1, y1, data1 = product_num(300, 2, 30, -30)
    x2, y2, data2 = product_num(300, 2, 10, 20)
    x3, y3, data3 = product_num(300, 2, 20, 50)
    x4, y4, data4 = product_num(300, 2, 20, 110)

    label1 = np.zeros(300, dtype=int)
    label2 = np.zeros(300, dtype=int) + 1
    label3 = np.zeros(300, dtype=int) + 2
    label4 = np.zeros(300, dtype=int) + 3

    data = np.concatenate((data1, data2, data3, data4), axis=0)
    label = np.concatenate((label1, label2, label3, label4), axis=0)

    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2, c='b')
    plt.scatter(x3, y3, c='g')
    plt.scatter(x4, y4, c='y')
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(data, label)

    model = LinearRegression()
    # model = Lasso()
    # model = LogisticRegression()
    model.fit(x_train, y_train)
    pred1 = model.predict(x_test)
    prediction = model.predict(data)
    prediction = np.array(prediction, dtype=int)

    np.set_printoptions(threshold=10000)
    print(prediction)
    # prediction = handle_pred_res(prediction)
    # print(prediction)
    # print(prediction.shape)

    print(model.score(x_test, y_test))
