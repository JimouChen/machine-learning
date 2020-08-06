"""
# @Time    :  2020/8/6
# @Author  :  Jimou Chen
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV  # 导入弹性网模型
import matplotlib.pyplot as plt

data = np.genfromtxt('longley.csv', delimiter=',')
x_data = data[1:, 2:]
y_data = data[1:, 1]

model = ElasticNetCV()  # 里面的系数会自动选出最好的，和lasso类似
model.fit(x_data, y_data)

# 预测对比某一行（第2行）
print(model.predict(x_data[2, np.newaxis]))

y_predict = []
for i in range(len(x_data)):
    y_predict.append(model.predict(x_data[i, np.newaxis]))

# 画图对比所有的
year = np.linspace(1947, 1962, len(x_data))
plt.plot(year, y_data, 'b.')
plt.plot(year, y_predict, 'r')
plt.show()

print('弹性网系数：', model.alpha_)
print('相关系数：', model.coef_)
