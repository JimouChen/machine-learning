"""
# @Time    :  2020/8/12
# @Author  :  Jimou Chen
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import BaggingClassifier  # bagging分类器
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

'''一般来说集成学习用于复杂的较好，下面是简单例子'''


# 定义一个画出两个特征的分布图，二维
def draw(model):
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


iris = datasets.load_iris()
x_data = iris.data[:, :2]  # 为了比对用了集成学习和不用集成学习的效果，只用两个特征
y_data = iris.target

# 切分数据
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

'''建kNN模型'''
kNN = KNeighborsClassifier()
kNN.fit(x_train, y_train)

# 画图
draw(kNN)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 准确率
print(kNN.score(x_test, y_test))

'''建决策树模型'''
tree = tree.DecisionTreeClassifier()
tree.fit(x_train, y_train)

# 画图
draw(tree)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 准确率
print(tree.score(x_test, y_test))

'''接下来使用bagging集成学习，加入kNN'''
# 100个不放回的抽样，也就是训练100个kNN分类器
bagging_kNN = BaggingClassifier(kNN, n_estimators=100)
bagging_kNN.fit(x_train, y_train)
# 画图
draw(bagging_kNN)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 准确率
print(bagging_kNN.score(x_test, y_test))

'''加入决策树的集成学习'''
bagging_tree = BaggingClassifier(tree, n_estimators=100)
bagging_tree.fit(x_train, y_train)

# 画图
draw(bagging_tree)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 准确率
print(bagging_tree.score(x_test, y_test))
