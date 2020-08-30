"""
# @Time    :  2020/8/21
# @Author  :  Jimou Chen
"""
from sklearn.neural_network import MLPClassifier  # 待会用神经网络预测降维后的数据
from sklearn.datasets import load_digits  # 手写数据集
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # 导入PCA模型
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

digits = load_digits()
x_data = digits.data  # 数据
y_data = digits.target  # 标签
# 切分数据
x_train, x_test, y_train, y_text = train_test_split(x_data, y_data)

# 建立神经网络模型,包含两个隐藏层，分别由100和50个神经元
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
model.fit(x_train, y_train)

# 评估
prediction = model.predict(x_test)
print(prediction)
print(classification_report(prediction, y_text))
print(confusion_matrix(prediction, y_text))
# 进行pca降维,这里n_components是降成2维
pca = PCA(n_components=2)
# 直接返回降维后的数据,如果不返回新数据，就用fit
new_data = pca.fit_transform(x_data)
print(new_data)

# 画出降维后的数据
new_x_data = new_data[:, 0]
new_y_data = new_data[:, 1]
plt.scatter(new_x_data, new_y_data, c='r')
plt.show()

# 这里的是只拟合，不返回重构的新数据
pca = model.fit(x_train, y_train)
pred = model.predict(x_data)
print(classification_report(pred, y_data))

# 画出预测的聚类图
plt.scatter(new_x_data, new_y_data, c=pred)
# plt.scatter(new_x_data, new_y_data, c=y_data)
plt.show()

'''降成3个维度'''


pca = PCA(n_components=3)
new_data = pca.fit_transform(x_data)
# print(new_data)
# 画出降维后的数据
new_x_data = new_data[:, 0]
new_y_data = new_data[:, 1]
new_z_data = new_data[:, 2]
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(new_x_data, new_y_data, new_z_data, c=pred, s=10)
# ax.scatter(new_x_data, new_y_data, new_z_data, c=y_data, s=10)
plt.show()
