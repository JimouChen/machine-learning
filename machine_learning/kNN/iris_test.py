"""
# @Time    :  2020/8/3
# @Author  :  Jimou Chen
"""

from sklearn import datasets  # 导入数据集
from sklearn.model_selection import train_test_split  # 切分数据
from sklearn.metrics import classification_report, confusion_matrix  # 验证准确性
import operator
import numpy as np


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


# 载入数据
iris = datasets.load_iris()
# 切分数据集, 0.2为测试集，0.8为训练集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

prediction = []
for i in range(x_test.shape[0]):
    prediction.append(kNN(x_test[i], x_train, y_train, 5))

# 拿测试的和预测的作比较，看看效果
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))
