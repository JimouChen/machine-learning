"""
# @Time    :  2020/8/13
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report

'''适用于数据多的情况，但是一般情况还是用KMeans'''

data = np.genfromtxt('kmeans.txt', delimiter=' ')
k = 4

model = MiniBatchKMeans(n_clusters=k)
model.fit(data)

# 分类中心坐标
centers = model.cluster_centers_
print(centers)

# 预测结果
pred_res = model.predict(data)
print(pred_res)

# 画图
colors = ['or', 'ob', 'og', 'oy']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], colors[pred_res[i]])

# 画出各个分类的中心点
mark = ['*r', '*b', '*g', '*y']
for i, center in enumerate(centers):
    plt.plot(center[0], center[1], mark[i], markersize=20)

plt.show()
