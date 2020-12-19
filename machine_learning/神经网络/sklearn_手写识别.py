"""
# @Time    :  2020/8/8
# @Author  :  Jimou Chen
"""
from sklearn.datasets import load_digits  # 导入数据集
from sklearn.neural_network import MLPClassifier  # 导入神经网络模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

digits = load_digits()
x_data = digits.data
y_data = digits.target
# print(x_data.shape)
# print(y_data)

# 切分数据
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

# 建模,2个隐藏层，第一个隐藏层有100个神经元，第2隐藏层80个神经元，训练500周期
model = MLPClassifier(hidden_layer_sizes=(100, 80), max_iter=500)
model.fit(x_train, y_train)

# 评估
prediction = model.predict(x_test)
print(classification_report(y_test, prediction))
print('origin：\n', y_test)
print('predict result:\n', prediction)
