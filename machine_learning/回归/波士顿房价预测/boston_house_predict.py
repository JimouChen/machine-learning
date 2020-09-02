"""
# @Time    :  2020/9/2
# @Author  :  Jimou Chen
"""
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd
import seaborn  # 这个是用来画系数相关性的热力图
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # 用于数据标准化
from sklearn.metrics import classification_report

house = load_boston()
# print(house.DESCR)

x = house.data
y = house.target

'''转成DataFrame是为了热力图可以使用到corr()方法'''
# 把标签和特征加在一起
df = pd.DataFrame(x, columns=house.feature_names)
# 加上标签列
df['Target'] = pd.DataFrame(y, columns=['Target'])
# print(df)

'''根据热力图找到标签列正系数较大的，即为影响较大的变量'''
# 画热力图， 比较两个变量之间的相关系数df.corr()
plt.figure(figsize=(15, 15))
pic = seaborn.heatmap(df.corr(), annot=True, square=True)
plt.show()

# 数据标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x)

# 切分数据
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3)

# 建模
model = LassoCV()
model.fit(x_train, y_train)

print('lasso系数', model.alpha_)
print('相关系数', model.coef_)

print('评估：', model.score(x_test, y_test))
