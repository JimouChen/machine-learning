"""
# @Time    :  2020/8/8
# @Author  :  Jimou Chen
"""
import operator
import numpy as np
import matplotlib.pyplot as plt


# 定义kNN函数，采用欧氏距离计算，返回预测的分类结果
def kNN(x_test, x_data, y_data, k):
    # 计算样本数量
    x_data_size = x_data.shape[0]
    # 复制x_test
    x_test_copy = np.tile(x_test, (x_data_size, 1))
    # 计算x_test与每个样本的差值
    diff_mat = x_test_copy - x_data
    # 计算差值平方
    sq_diff_mat = diff_mat ** 2
    # 求和
    sq_distance = sq_diff_mat.sum(axis=1)
    # 开方,得到每个样本与测试样本的距离
    distance = sq_distance ** 0.5
    # 从小到大排序
    sorted_distance = distance.argsort()
    # 进行分类，把分类结果按多到少放到一个字典
    class_count = {}
    for i in range(k):
        # 获取标签
        label = y_data[sorted_distance[i]]
        # 统计标签数量
        class_count[label] = class_count.get(label, 0) + 1
    # 将分类结果从数量按多到少排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


x_data = np.array([[1.0, 1.1],
                   [2.0, 2.0],
                   [0, 0],
                   [4.1, 5.1]])

y_data = ['A', 'B', 'C', 'D']
plt.plot(x_data[:, 0], x_data[:, 1], 'b.')
plt.show()
print(kNN([1, 1], x_data, y_data, 3))
