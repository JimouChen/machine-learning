"""
# @Time    :  2020/8/12
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier  # 导入AdaBoost模型
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import classification_report


# 定义一个画出两个特征的分布图，二维，只适用于两个特征
def draw(model, x_data, y_data):
    # 获取数值所在范围
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # 等高线图
    cs = plt.contourf(xx, yy, z)
    # 样本散点图
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    plt.show()


# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征
x1, y1 = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征均值都为3
x2, y2 = make_gaussian_quantiles(mean=(3, 3), n_samples=500, n_features=2, n_classes=2)

# 将两组数据合成一组数据, 也就是得到1000个样本
x_data = np.concatenate((x1, x2))
y_data = np.concatenate((y1, - y2 + 1))

plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

'''使用决策树看看结果'''
d_tree = tree.DecisionTreeClassifier(max_depth=3)
d_tree.fit(x_data, y_data)

# 画图预测
draw(d_tree, x_data, y_data)
print(d_tree.score(x_data, y_data))

'''使用AdaBoost模型，看看效果'''
AdaBoost = AdaBoostClassifier(d_tree, n_estimators=10)
AdaBoost.fit(x_data, y_data)

# 画图
draw(AdaBoost, x_data, y_data)
print(AdaBoost.score(x_data, y_data))
