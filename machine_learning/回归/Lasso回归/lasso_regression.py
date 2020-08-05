"""
# @Time    :  2020/8/5
# @Author  :  Jimou Chen
"""
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

data = np.genfromtxt('longley.csv', delimiter=',')
x_data = data[1:, 2:]
y_data = data[1:, 1]

# 建立模型拟合,这里用的交叉验证的lasso
model = LassoCV()
model.fit(x_data, y_data)

# lasso系数
print('lasso系数: ', model.alpha_)
# 相关系数
print('lasso相关系数：', model.coef_)

# 预测测试
print('原始数据：', y_data, end='  ')
# print(model.predict(x_data[-2, np.newaxis]))
# 预测数据
y_predict_data = []
for i in range(len(x_data)):
    i_predict_data = model.predict(x_data[i, np.newaxis])[0]
    y_predict_data.append(i_predict_data)
    # print(model.predict(x_data[i, np.newaxis]))

print('\n', y_predict_data)
