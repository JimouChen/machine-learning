"""
# @Time    :  2020/8/12
# @Author  :  Jimou Chen
"""
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier  # 导入Voting

'''
voting的建模步骤和stacking的建模过程类似，但是没有次级分类器
'''
# 载入数据集
iris = datasets.load_iris()
# 只要第1,2列的特征
x_data, y_data = iris.data[:, 1:3], iris.target

# 定义三个不同的分类器
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()

# 建立模型，按照这个格式建立Voting模型,
# 每个元组的第一个是对分类器的描述
voting_model = VotingClassifier([('kNN', clf1), ('d_tree', clf2), ('lr', clf3)])

for clf, label in zip([clf1, clf2, clf3, voting_model],
                      ['KNN', 'Decision Tree', 'LogisticRegression', 'VotingClassifier']):
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))
