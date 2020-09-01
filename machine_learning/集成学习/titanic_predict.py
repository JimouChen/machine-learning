"""
# @Time    :  2020/9/1
# @Author  :  Jimou Chen
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')
# print(data)

'''先处理空缺的数据'''

# 处理空缺的年龄，设为平均年龄
data['Age'] = data['Age'].fillna(data['Age'].median())
# print(data.describe())
# 处理性别，转化维0和1,loc是取数据的，里面传行，列
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0
# print(data.loc[:, 'Sex'])

# 处理Embarked，登录港口
# print(data['Embarked'].unique())  # 看一下里面有几类
# 由于'S'比较多，就把空值用S填充
data['Embarked'] = data['Embarked'].fillna('S')
# 转化为数字
data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

'''接下来选取有用的特征'''
feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_data = data[feature]
y_data = data['Survived']  # 预测的标签

# 数据标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
# print(x_data)

'''处理完数据之后，现在可以使用各自算法看看效果了'''

from sklearn.model_selection import cross_val_score  # 导入交叉验证后的分数
# 逻辑回归
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
# 计算交叉验证的误差，分三组
scores = cross_val_score(lr, x_data, y_data, cv=3)
print(scores.mean())  # 求平均

# 神经网络模型
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=2000)
# 计算交叉验证的误差，分三组
scores = cross_val_score(mlp, x_data, y_data, cv=3)
print(scores.mean())  # 求平均

# kNN
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=21)
scores = cross_val_score(kNN, x_data, y_data, cv=3)
print(scores.mean())

# 决策树
from sklearn.tree import DecisionTreeClassifier

# 最小分割样本数，小于4个就不往下分割了
d_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=4)
scores = cross_val_score(d_tree, x_data, y_data, cv=3)
print(scores.mean())

'''下面是集成学习'''
# 随机森林
from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2)
scores = cross_val_score(rf1, x_data, y_data, cv=3)
print(scores.mean())

# 100棵决策树构成
rf2 = RandomForestClassifier(n_estimators=100, min_samples_split=4)
scores = cross_val_score(rf2, x_data, y_data, cv=3)
print(scores.mean())

# Bagging
from sklearn.ensemble import BaggingClassifier

# 集成rf2，做20次有放回的抽样,由于rf2也是集成学习模型，所以运行时间有点久
bg = BaggingClassifier(rf2, n_estimators=20)
scores = cross_val_score(bg, x_data, y_data, cv=3)
print(scores.mean())

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier(rf2, n_estimators=20)
scores = cross_val_score(adb, x_data, y_data, cv=3)
print(scores.mean())

# Stacking
from mlxtend.classifier import StackingClassifier

stacking = StackingClassifier(classifiers=[bg, mlp, lr],
                              meta_classifier=LogisticRegression())
scores = cross_val_score(stacking, x_data, y_data, cv=3)
print(scores.mean())

# Voting
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier([('ado', adb), ('mlp', mlp),
                           ('LR', lr), ('kNN', kNN),
                           ('d_tree', d_tree)])
scores = cross_val_score(voting, x_data, y_data, cv=3)
print(scores.mean())
