"""
# @Time    :  2020/8/30
# @Author  :  Jimou Chen
"""
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林模型


data = pd.read_csv('data/wine.csv')
y_data = data.iloc[:, 0]
x_data = data.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
pca = PCA(n_components=2)
new_data = pca.fit_transform(x_data)

# 画出来看一下
plt.scatter(new_data[:, 0], new_data[:, 1], c=y_data)
plt.show()

# 建模预测
# model = svm.SVC(decision_function_shape='ovo')
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

prediction = model.predict(x_data)

print(model.score(x_test, y_test))
print(classification_report(y_data, prediction))

# 画出预测的
plt.scatter(new_data[:, 0], new_data[0:, 1], c=prediction)
plt.show()

#
# # 画3d图
# z_data = new_data[:, 2]
# ax = plt.figure().add_subplot(111, projection='3d')
# ax.scatter(new_data[:, 0], new_data[:, 1], z_data, c=y_data, s=10)  # s是大小
# plt.show()
#
# ax = plt.figure().add_subplot(111, projection='3d')
# ax.scatter(new_data[:, 0], new_data[:, 1], z_data, c=prediction, s=10)  # s是大小
# plt.show()