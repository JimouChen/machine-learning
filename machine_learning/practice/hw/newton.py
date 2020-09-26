"""
# @Time    :  2020/9/26
# @Author  :  Jimou Chen
"""


def f(a, b, c, x):
    return a * (x ** 2) + b * x + c


# f(x)的一阶导数
def f_(a, b, x):
    return 2 * a * x + b


# f(x)的二阶导数
def f__(a, x):
    return 2 * a


# x是一开始随便给的初始值
def newton_update_x(a, b, x, epochs):
    for i in range(epochs):
        # 每次更新函数值和导数值
        func_ = f_(a, b, x)
        func__ = f__(a, x)
        temp = x
        # 如果导数为0，说明已经找到最值
        if func_ == 0:
            return x
        # 更新与x轴的交点横坐标x
        x = x - func_ / func__
        # 当x和上一次的x很接近时，说明已经找到该一元二次方程的最值点处
        if abs(x - temp) < 1e-7:
            return x


if __name__ == '__main__':
    # 输入a，b，c
    a, b, c = map(int, input().split())
    random_x = 4
    res_x = newton_update_x(a, b, random_x, 2)
    print('牛顿法求得的最值处的坐标是: (%.3f, %.3f)' % (res_x, f(a, b, c, res_x)))
    print('求得的最值是:%.3f' % f(a, b, c, res_x))
