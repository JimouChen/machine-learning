"""
# @Time    :  2020/8/3
# @Author  :  Jimou Chen
"""
import operator
import numpy as np


# 定义knn函数，采用欧氏距离计算，返回预测的分类结果
def kNN(x_test, x_data, y_data, k):
    sorted_distance = ((((np.tile(x_test, (x_data.shape[0], 1)) - x_data) ** 2).sum(axis=1)) ** 0.5).argsort()
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
