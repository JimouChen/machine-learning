"""
# @Time    :  2020/8/8
# @Author  :  Jimou Chen
"""
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 读入数据
iris = load_iris()
x_data = iris.data
y_data = iris.target

# 切分数据
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# 建模,n_neighbors即为k
# kNN_model = KNeighborsClassifier()
kNN_model = KNeighborsClassifier(n_neighbors=5)
kNN_model.fit(x_train, y_train)

predictions = kNN_model.predict(x_test)
print('origin: \n', y_test)
print('predict result:\n', predictions)
print(classification_report(y_test, predictions))

# 调用该对象的打分方法，计算出准确率
# print(kNN_model.score(x_test, y_test, sample_weight=None))
print(kNN_model.score(x_test, y_test))
