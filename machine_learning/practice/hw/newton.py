"""
# @Time    :  2020/9/26
# @Author  :  Jimou Chen
"""


# 更新b和k
def get_bk(x_data, y_data, b, k, lr, epochs):
    # 计算总数据量
    m = len(x_data)
    # 循环epochs次
    for i in range(epochs):
        # 临时变量
        b_grad = 0
        k_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, m):
            b_grad += (1 / m) * (k * x_data[j] + b - y_data[j])
            k_grad += (1 / m) * (k * x_data[j] + b - y_data[j]) * x_data[j]
        # 更新b，k
        b = b - lr * b_grad
        k = k - lr * k_grad

    return b, k


def f(x):
    return (x - 2) ** 3-3*x


# f(x)的导数
def f_(x):
    return 3 * ((x - 2) ** 2) - 3


# x是一开始随便给的初始值
def newton_update_x(x, epochs):
    for i in range(epochs):
        # 每次更新函数值和导数值
        func = f(x)
        func_ = f_(x)

        # 更新与x轴的交点横坐标x
        x = x - func / func_
        print(x)
        # 当上一次的y值和这次的更新的y值很接近时，x也很接近于零点附近
        if func - f(x) < 1e-7:
            return x


if __name__ == '__main__':
    res = newton_update_x(4, 100)
    print('牛顿法求得的解为: x =', res)