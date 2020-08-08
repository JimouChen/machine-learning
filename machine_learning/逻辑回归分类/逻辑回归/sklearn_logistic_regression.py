"""
# @Time    :  2020/8/6
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import classification_report  # 用来对模型的预测效果做评估

data = np.genfromtxt('LR-testSet.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]


# 切分有效数据，可以画出散点图
def show_scatter():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])

    # plt.plot(x0, y0, 'b.')
    # plt.plot(x1, y1, 'rx')
    # plt.show()
    # 画出散点图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    # 画图例
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')


# 先把散点图画出来看看
show_scatter()
plt.show()
# 建模，拟合
model = LogisticRegression()
model.fit(x_data, y_data)

# 画出决策边界,数据无标准化
show_scatter()
x_test = np.array([[-4], [3]])
y_test = (-model.intercept_ - x_test * model.coef_[0][0]) / model.coef_[0][1]
plt.plot(x_test, y_test, 'k')
plt.show()

# 如果要把这个模型拿来预测，可以这样做
prediction = model.predict(x_data)
print(prediction)

# 对预测模型进行评估
print(classification_report(y_data, prediction))
