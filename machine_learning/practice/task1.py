"""
# @Time    :  2020/8/22
# @Author  :  Jimou Chen
"""

'''
1.读取 wine 数据集，区分标签和数据。
2.将 wine 数据集划分为训练集和测试集。
3.使用离差标准化方法标准化 wine 数据集。
4.构建 SVM 模型，并预测测试集结果。
5.打印出分类报告，评价分类模型性能。
'''

from sklearn.preprocessing import MinMaxScaler  # 用于数据标准化
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd

wine_data = pd.read_csv('data/wine.csv' )

x_data = wine_data.iloc[:, 1:]
y_data = wine_data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)

# 数据标准化
std = MinMaxScaler().fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

# 建模
model = svm.SVC()
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(pred)

y_test = y_test.values
print(y_test)

print(classification_report(pred, y_test))
print(model.score(x_test, y_test))
