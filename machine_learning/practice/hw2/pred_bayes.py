"""
# @Time    :  2020/10/8
# @Author  :  Jimou Chen
"""
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import pandas as pd


# 处理训练数据
def handle_data(data):
    # 将非数值的特征转换为数值
    data.loc[data['Outlook'] == 'Sunny', 'Outlook'] = 0
    data.loc[data['Outlook'] == 'Overcast', 'Outlook'] = 1
    data.loc[data['Outlook'] == 'Rain', 'Outlook'] = 2

    data.loc[data['Temperature'] == 'Hot', 'Temperature'] = 0
    data.loc[data['Temperature'] == 'Mild', 'Temperature'] = 1
    data.loc[data['Temperature'] == 'Cool', 'Temperature'] = 2

    data.loc[data['Humidity'] == 'High', 'Humidity'] = 0
    data.loc[data['Humidity'] == 'Normal', 'Humidity'] = 1

    data.loc[data['Wind'] == 'Weak', 'Wind'] = 0
    data.loc[data['Wind'] == 'Strong', 'Wind'] = 1

    # 处理标签，转化为数值
    data.loc[data['PlayTennis'] == 'No', 'PlayTennis'] = 0
    data.loc[data['PlayTennis'] == 'Yes', 'PlayTennis'] = 1

    # 返回处理后的训练集和测试数据
    x_data = data.iloc[:-1, 1:-1]
    y_data = data.iloc[:-1, -1].astype('int')
    x_test = data.iloc[-1, 1:-1].values.reshape(1, -1)

    return x_data, y_data, x_test


if __name__ == '__main__':
    my_data = pd.read_csv('data.csv')
    x_data, y_data, x_test = handle_data(my_data)

    # 建模拟合，这里使用高斯模型
    model = GaussianNB()
    # model = BernoulliNB()  # 发现伯努利模型效果更好
    model.fit(x_data, y_data.ravel())

    print('准确率:', model.score(x_data, y_data))
    prediction = model.predict(x_test)
    print('预测的类别是:', prediction)

    if prediction == [0]:
        print('No, 不去打球')
    else:
        print('Yes， 去打球')
