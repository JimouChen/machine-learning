"""
# @Time    :  2020/9/26
# @Author  :  Jimou Chen
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# 更新b和k
def update_bk(x_data, y_data, b, k, lr, epochs):
    # 计算总数据量
    m = len(x_data)
    # 循环epochs次
    for i in range(epochs):
        # 临时变量
        b_grad = 0
        k_grad = 0
        # 计算梯度的总和再求平均,这样精确一点
        for j in range(0, m):
            b_grad += (1 / m) * (k * x_data[j] + b - y_data[j])
            k_grad += (1 / m) * (k * x_data[j] + b - y_data[j]) * x_data[j]
        # 更新b，k
        b = b - lr * b_grad
        k = k - lr * k_grad

    return b, k


if __name__ == '__main__':
    # 生成数据，加入少量噪点
    x_data, y_data = make_regression(n_samples=200, noise=15, n_features=1, random_state=20)

    # 设置学习率learning rate,截距，斜率
    lr = 0.03
    b = 0
    k = 0
    # 最大迭代次数
    epochs = 200

    b, k = update_bk(x_data, y_data, b, k, lr, epochs)
    print('迭代后， b = {}, k = {}'.format(b, k))

    # 画出图像
    plt.plot(x_data, y_data, 'b.')
    # 画出回归直线,在同一张图上画
    plt.plot(x_data, k * x_data + b, 'r')
    plt.show()
