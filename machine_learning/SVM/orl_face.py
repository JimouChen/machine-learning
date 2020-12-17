"""
# @Time    :  2020/12/7
# @Author  :  Jimou Chen
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

if __name__ == '__main__':
    face = fetch_olivetti_faces(shuffle=True)
    print(face.images.shape)
    x_data = face.data
    y_data = face.target
    # print(x_data.shape)
    # print(y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=20)

    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    # pred = model.predict(x_test)
    # print(classification_report(pred, y_test))

    '''use pca'''
    # 上面64*64个维度降成50个维度
    pca = PCA(n_components=20, whiten=True).fit(face.data)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    pca_model = SVC(kernel='rbf')
    pca_model.fit(x_train, y_train)

    print(pca_model.score(x_test, y_test))
    # pred = pca_model.predict(x_test)
