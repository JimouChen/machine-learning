"""
# @Time    :  2020/8/13
# @Author  :  Jimou Chen
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB  # 导入朴素贝叶斯的三种模型

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

'''建立三种贝叶斯模型看看效果'''

# 建立多项式模型
mul = MultinomialNB()
mul.fit(x_train, y_train)
print(classification_report(mul.predict(x_test), y_test))
print(confusion_matrix(mul.predict(x_test), y_test))

# 建立伯努利模型
bernoulli = BernoulliNB()
bernoulli.fit(x_train, y_train)
print(classification_report(bernoulli.predict(x_test), y_test))
print(confusion_matrix(bernoulli.predict(x_test), y_test))

# 建立高斯模型
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
print(classification_report(gaussian.predict(x_test), y_test))
print(confusion_matrix(gaussian.predict(x_test), y_test))
