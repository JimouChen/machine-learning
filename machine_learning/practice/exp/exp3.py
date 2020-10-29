"""
# @Time    :  2020/10/29
# @Author  :  Jimou Chen
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np

if __name__ == '__main__':
    # 读入数据
    data = np.genfromtxt('dtree.csv', delimiter=',')
    x_data = data[1: -1, :-1]
    y_data = data[1:-1, -1].reshape(-1, 1)
    x_test = data[-1, :-1].reshape(1, -1)

    # 建立决策树模型
    model = DecisionTreeClassifier()
    model.fit(x_data, y_data)
    prediction = model.predict(x_test)
    print('预测结果:', prediction)
    if int(prediction[0] == 0):
        print('通过预测，测试数据的生活水平为: 低')
    else:
        print('通过预测，测试数据的生活水平为: 高')
