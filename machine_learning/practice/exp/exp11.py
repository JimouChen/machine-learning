"""
# @Time    :  2020/10/15
# @Author  :  Jimou Chen
"""
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 通过计算斜率和截距画拟合直线
def draw_line(model, color):
    k = model.coef_
    b = model.intercept_
    line = k * x_data + b
    plt.plot(x_data, line, color, 50)
    plt.scatter(x_data, y_data, c='r')
    plt.show()


# 获取表达式
def get_express(model):
    k = '%.2f' % model.coef_[0]
    b = model.intercept_
    if b > 0:
        b = '+' + '%.2f' % b
    else:
        b = '%.2f' % b
    return 'y = ' + k + '*x' + b


if __name__ == '__main__':
    # 产生一组数据, 加入噪点
    x_data, y_data = make_regression(n_samples=200, noise=20, n_features=1, random_state=30)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    y_data = y_data.reshape(-1, 1)

    # 线性回归
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    lr_prediction = lr_model.predict(x_test)
    print('线性回归准确率:', lr_model.score(x_test, y_test))
    print('线性回归的表达式是:', get_express(lr_model))

    # 岭回归
    rd_model = Ridge()
    rd_model.fit(x_train, y_train)
    rd_prediction = rd_model.predict(x_test)
    print('岭回归准确率:', rd_model.score(x_test, y_test))
    print('岭回归的表达式是:', get_express(rd_model))

    # 套索回归
    ls_model = Lasso()
    ls_model.fit(x_train, y_train)
    ls_prediction = ls_model.predict(x_test)
    print('套索回归准确率:', ls_model.score(x_test, y_test))
    print('套索回归的表达式是:', get_express(ls_model))

    draw_line(lr_model, 'b')
    draw_line(rd_model, 'y')
    draw_line(ls_model, 'g')
