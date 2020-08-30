"""
# @Time    :  2020/8/30
# @Author  :  Jimou Chen
"""
# 求取原始特征之间的相关系数
import numpy as np
import pandas as pd


# 定义一个相关性的热力图，更加直观地判断
def heat_map(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(data.shape[0], data.shape[1]))  # 尺寸大小与data一样
    correlation_mat = data.corr()
    sns.heatmap(correlation_mat, annot=True, cbar=True, square=True, fmt='.2f', annot_kws={'size': 10})
    plt.show()


data = pd.read_csv('data/wine.csv')
data = data.iloc[:, 1:]  # 年份去掉
# 计算列与列之间的相关系数，返回相关系数矩阵，保留3位小数
print('相关系数矩阵：\n', np.round(data.corr(method='pearson'), 3))
# 作出相关性热力图
heat_map(data)
