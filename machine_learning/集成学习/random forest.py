"""
# @Time    :  2020/8/12
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林模型
from sklearn import tree  # 导入决策树模型，与随机森林模型做对比


# 定义一个画预测图
def draw(model):
    # 获取数据值所在的范围
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = z.reshape(xx.shape)
    # 等高线图
    cs = plt.contourf(xx, yy, z)
    # 样本散点图
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    plt.show()


data = np.genfromtxt('LR-testSet2.txt', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]

plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 切分数据
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

# 建立决策树模型
tree = tree.DecisionTreeClassifier()
tree.fit(x_train, y_train)
draw(tree)
# 评估模型
print(tree.score(x_test, y_test))

# 建立随机森林模型
RF = RandomForestClassifier(n_estimators=100)
RF.fit(x_train, y_train)
draw(RF)
# 评估模型
print(RF.score(x_test, y_test))
