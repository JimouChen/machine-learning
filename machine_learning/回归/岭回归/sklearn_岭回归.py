"""
# @Time    :  2020/8/5
# @Author  :  Jimou Chen
"""
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

'''
预测GNP.deflator
数据有多重共线性
'''

data = np.genfromtxt(r'longley.csv', delimiter=',')
x_data = data[1:, 2:]
y_data = data[1:, 1]

# 不指定生成几个数，默认生成50个值
# 待会拿这些值来测试看看哪个值最适合做岭系数λ，这里用α表示
alphas_test = np.linspace(0.001, 1)
# 创建模型，保存误差值,Right是岭回归，CV是交叉验证
# 它会自动测试50个值，并自动选出最好的系数
model = linear_model.RidgeCV(alphas=alphas_test, store_cv_values=True)
model.fit(x_data, y_data)

# 岭系数
print('最好的岭系数', model.alpha_)
# loss值
# print(model.cv_values_)

# 画图
# 岭系数与loss值的关系
# mean是求50个loss值的平均值，axis=0是每行
plt.plot(alphas_test, model.cv_values_.mean(axis=0))
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
plt.show()

# 现在对每一行做预测，做对比
print(y_data)
for i in range(len(x_data)):
    print(model.predict(x_data[i, np.newaxis]), end='  ')
