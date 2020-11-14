"""
# @Time    :  2020/11/14
# @Author  :  Jimou Chen
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd


def kmeans_cluster(data, n):
    model = KMeans(n_clusters=n)
    model.fit(data)
    label = model.labels_
    score = silhouette_score(data, label)
    return score


if __name__ == '__main__':

    # input:data(ndarray):样本数据
    # output:3个轮廓函数的值(float)
    # ********* Begin *********#
    data = pd.read_csv('dateset.csv')
    data = data.iloc[0:, 1:]

    # 取n = 3，4，7
    score1 = kmeans_cluster(data, 3)
    print('聚成3类时的轮廓评价函数值:', score1)
    score2 = kmeans_cluster(data, 4)
    print('聚成4类时的轮廓评价函数值:', score2)
    score3 = kmeans_cluster(data, 7)
    print('聚成7类时的轮廓评价函数值:', score3)

    # 画出轮廓评价函数值曲线变化图，2-10类
    x_range = [i for i in range(2, 11)]
    y_score = []
    for i in range(2, 11):
        y_score.append(kmeans_cluster(data, i))
    plt.plot(x_range, y_score, 'o-', color='b')
    plt.xlabel('n clusters')
    plt.ylabel('evaluation score')
    plt.show()
    # ********* End *********#
