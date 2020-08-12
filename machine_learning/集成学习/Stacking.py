"""
# @Time    :  2020/8/12
# @Author  :  Jimou Chen
"""
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier  # 导入Stacking模型分类器

# 载入数据集
iris = datasets.load_iris()
# 只要第1,2列的特征
x_data, y_data = iris.data[:, 1:3], iris.target

# 定义三个不同的分类器
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()

# 定义一个次级分类器 meta_classifier, 传入的是逻辑回归模型
lr = LogisticRegression()
stacking_model = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, stacking_model],
                      ['KNN', 'Decision Tree', 'LogisticRegression', 'StackingClassifier']):
    # cross_val_score进行交叉验证， cv=3是分3份进行交叉验证，scoring='accuracy'计算准确率
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=3, scoring='accuracy')
    # scores.mean()求三个交叉验证结果的平均值
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))
