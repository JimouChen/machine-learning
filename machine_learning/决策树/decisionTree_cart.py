"""
# @Time    :  2020/8/12
# @Author  :  Jimou Chen
"""
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report

data = np.genfromtxt('cart.csv', delimiter=',')
x_data = data[1:, 1: -1]
y_data = data[1:, -1]

# 不带参数默认是jini系数，也就是cart算法
model = tree.DecisionTreeClassifier()
model.fit(x_data, y_data)

prediction = model.predict(x_data)
print('origin:\n', y_data)
print('predict data:\n', prediction)

print(classification_report(prediction, y_data))