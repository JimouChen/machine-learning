"""
# @Time    :  2020/9/26
# @Author  :  Jimou Chen
"""
import operator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def kNN(x_test, x_data, y_data, k):
    # 计算多少个样本
    x_data_size = x_data.shape[0]
    # 复制x_test，复制行x_data_size次，复制列1次
    x_test_copy = np.tile(x_test, (x_data_size, 1))
    # 计算x_test与每个样本的差值 ,计算差值平方
    sq_diff_mat = (x_test_copy - x_data) ** 2
    # 对差值求和,再开方,得到每个样本与测试样本的距离
    distance = (sq_diff_mat.sum(axis=1)) ** 0.5
    # 对上面得到的距离索引进行从小到大排序
    sorted_distance = distance.argsort()
    # 进行分类，把分类结果按多到少放到一个字典
    class_count = {}
    for i in range(k):
        # 获取标签
        label = y_data[sorted_distance[i]]
        # 统计标签数量，如果不存在该标签就把数量置0加1
        class_count[label] = class_count.get(label, 0) + 1
    # 将分类结果按数量按多到少排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # 返回数量最多的类别
    return sorted_class_count[0][0]


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    x_test = np.array([5, 5])
    y_test = kNN(x_test, x_data, y_data, 3)
    print('测试点识别的类别是:', y_test)

    plt.scatter(x_data.iloc[:, 0], x_data.iloc[:, 1], s=100)
    plt.scatter(x_test[0], x_test[1], c='r', s=100)
    plt.show()
