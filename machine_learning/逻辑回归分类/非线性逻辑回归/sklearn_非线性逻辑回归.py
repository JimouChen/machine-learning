"""
# @Time    :  2020/8/7
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_gaussian_quantiles  # 用于产生数据集
from sklearn.preprocessing import PolynomialFeatures

# 生成数据集，生成的是2维正态分布，可以自己设置类别
# 这里设为500个样本，2个样本特征，类别是2类，也可以设为多累
x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)

# 可以画出来看看,分类传到颜色c
plt.scatter(x_data[:, 0], x_data[0:, 1], c=y_data)
plt.show()

'''因为是非线性的，所以需要产生非线性特征'''

# 定义多项式回归，用degree来调节多项式特征
poly_reg = PolynomialFeatures(degree=5)
# 特征处理
x_poly = poly_reg.fit_transform(x_data)
# 建模拟合
model = LogisticRegression()
model.fit(x_poly, y_data)

# 评估
print('score: ', model.score(x_poly, y_data))
# 预测测试
print('原来的分类结果：\n', y_data)
print('预测第5行的结果是', model.predict(x_poly)[5])
print('所有的预测结果： \n', model.predict(x_poly))

'''上面已经建好模型了，可以直接去预测了，接下来是画图'''

# 获取数据值所在的范围,这里是确定图的边框范围
x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

# 生成网格矩阵，即很多点构成的背景图，尽量密集些
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测,ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
z = model.predict(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
z = z.reshape(xx.shape)

# 等高线图
cs = plt.contourf(xx, yy, z)
# 散点图
plt.scatter(x_data[:, 0], x_data[0:, 1], c=y_data)
plt.show()
