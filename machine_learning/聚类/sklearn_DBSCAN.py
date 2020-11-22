"""
# @Time    :  2020/8/21
# @Author  :  Jimou Chen
"""
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('kmeans.txt', delimiter=' ')
# 建模
# esp距离阈值，min_samples在esp领域里面的样本数,
# 也就是超过min_sample就可以当成一个类
model = DBSCAN(eps=1.5, min_samples=4)
model.fit(data)

# 预测,fit_predict是先拟合后预测，DBSCAN没有predict方法
pred = model.fit_predict(data)
print(pred)  # 预测值为-1的是噪点

# 画出各个数据点
mark = ['or', 'ob', 'og', 'oy', 'ok', 'om']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], mark[pred[i]])
plt.show()
