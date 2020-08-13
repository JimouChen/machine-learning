"""
# @Time    :  2020/8/13
# @Author  :  Jimou Chen
"""
# from sklearn.datasets import fetch_20newsgroups  # 导入新闻包
# from sklearn.model_selection import train_test_split
#
# # 导入数据，选all是导入所有，可以选train等
# news = fetch_20newsgroups(subset='train')
#
# print(news.target_names)
# print(len(news.data))
# print(len(news.target))
# print(news.data[0])

'''词袋模型'''
from sklearn.feature_extraction.text import CountVectorizer

texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())

print(cv_fit.toarray().sum(axis=0))
