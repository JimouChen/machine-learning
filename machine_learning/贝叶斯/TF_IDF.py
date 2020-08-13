"""
# @Time    :  2020/8/13
# @Author  :  Jimou Chen
"""
from sklearn.feature_extraction.text import TfidfVectorizer

# 文本文档列表
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
# 创建变换函数
vectorizer = TfidfVectorizer()
# 词条化以及创建词汇表
vectorizer.fit(text)
# 总结
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# 编码文档
vector = vectorizer.transform([text[0]])
# 总结编码文档
print(vector.shape)
print(vector.toarray())
