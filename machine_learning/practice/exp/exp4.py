# encoding=utf8
import os
import random
import numpy as np

# 实现方向传播算法
# ********* Begin *********#
# 获取训练数据
import pandas as pd
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv('./train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./train_label.csv')
train_label = train_label['target']

# 获取测试数据
test_data = pd.read_csv('./test_data.csv')
# 构建神经网络模型并训练
model = MLPClassifier(hidden_layer_sizes=(100, 80), max_iter=55, random_state=20)
model.fit(train_data, train_label)
# 获取预测值
prediction = model.predict(test_data)
# 转化为DataFrame格式方便保存
prediction = pd.DataFrame(prediction)
prediction.columns = ['result']  # 添加列标签名
prediction.to_csv('result.csv')  # 保存成csv文件
# ********* End *********#


# 获取预测标签
df_result = pd.read_csv('./result.csv')
predict = df_result['result']
# 获取真实标签
df_label = pd.read_csv('./test_label.csv')
label = df_label['target']
# 计算正确率
acc = np.mean(predict == label)

if acc > 0.95:
    print('预测准确率为:', acc)
    print('预测正确率高于0.95')
else:
    print('模型正确率为：%.3f,可以尝试修改' % acc)
