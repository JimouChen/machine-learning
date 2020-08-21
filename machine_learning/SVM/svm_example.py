"""
# @Time    :  2020/8/21
# @Author  :  Jimou Chen
"""
from sklearn.svm import SVC  # 导入svm的分类器SVC

# 二维的情况
x = [[3, 3], [4, 3], [1, 1]]
y = [1, 1, 0]  # 分类标签，2个类

# 建模, 核函数为线性
model = SVC(kernel='linear')
model.fit(x, y)

# 打印支持向量
print('支持向量：\n', model.support_vectors_)
# 看看哪几个点是支持向量，打印出来是第2和第0个
print('第几个点是支持向量：\n', model.support_)
# 支持向量的分布情况，在分界线两端,这里打出来是各有1个
print('支持向量在分界线两端的分布情况\n', model.n_support_)

# 预测类别
print('预测坐标(%d, %d)的类别是：' % (-8, 3), model.predict([[-8, 3]]))
print('预测坐标(%d, %d)的类别是：' % (4, 3), model.predict([[4, 3]]))

# 看看系数和截距
print('系数：\n', model.coef_)
print('截距:\n', model.intercept_)
