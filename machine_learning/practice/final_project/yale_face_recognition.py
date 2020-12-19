"""
# @Time    :  2020/12/19
# @Author  :  Jimou Chen
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# 得到模型的评估指标，F1-分数，召回率，ROC曲线
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, recall_score


class FaceRecognition:
    # 初始化参数
    def __init__(self, photo_path, save_file='yale_data.txt'):
        """
        :param photo_path: 图片路径
        :param save_file: 将图片转化为二维数据的文件名
        """
        self.path = photo_path
        self.save_file = save_file
        self.y_test = None
        self.y_predict = None
        self.model = None  # 保存最终训练得到的模型

    # 处理数据，将图片数据转化为二维矩阵
    def handle_data(self):
        # 标签列添加到矩阵的最后一列
        label_list = []
        # 将每一行的特征向量进行堆叠,最后得到(165,10000)大小的二维特征矩阵
        stack_matrix = np.array([[0]])
        for i in range(1, 16):
            # 加入每张图片的标签
            label_list.append(i)
            class_matrix = np.array(label_list, ndmin=2)
            for j in range(1, 12):
                self.path = photo_path.format(i, j)
                x = Image.open(self.path)
                # 转换为narray的结构，并转为二维矩阵
                data = np.reshape(np.asarray(x), (1, -1))
                # print(x_data.shape)  # 得到的维度是(1, 10304)
                one_data = np.column_stack((data, class_matrix))
                # 第一次不合并
                if i == 1 and j == 1:
                    stack_matrix = one_data
                    continue
                stack_matrix = np.row_stack((stack_matrix, one_data))

            label_list.pop()
        np.savetxt(self.save_file, stack_matrix)

    # 加载读入数据
    def load_data(self):
        file = self.save_file
        # 读入处理后的图片二维矩阵文件
        train_data = np.loadtxt(file)
        data = train_data[:, :10000]  # 取出特征数据
        target = train_data[:, -1]  # 取出标签数据
        return data, target

    # 训练模型，返回准确率和模型，并打印出F1-分数和召回率等评估参数
    def train_model(self, n_components=50, random_state=13):
        """
        :param n_components: PCA降维的维度
        :param random_state: 设置随机种子，调整后得到最佳模型
        :return: 返回准确率和模型
        """
        x_data, y_data = self.load_data()
        x_train, x_test, y_train, self.y_test = train_test_split(x_data, y_data,
                                                                 test_size=0.325,
                                                                 random_state=random_state)

        # 利用PCA将特征降至50维
        pca = PCA(n_components=n_components, whiten=True)
        x_train = pca.fit_transform(x_train)
        self.model = SVC(kernel='rbf', C=50)  # C是惩罚参数
        self.model.fit(x_train, y_train)

        # 利用在训练集上进行降维的PCA对测试数据进行降维，保证转换矩阵相同
        x_test_pca = pca.transform(x_test)
        self.y_predict = self.model.predict(x_test_pca)
        score = self.model.score(x_test_pca, self.y_test)
        print(classification_report(self.y_test, self.y_predict))
        return score, self.model

    # 画ROC图
    def draw_ROC(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_predict, pos_label=15)
        roc_auc = auc(fpr, tpr)
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()

    # 返回模型评估参数
    def model_evaluation(self):
        print('recall: %.4f' % recall_score(self.y_test, self.y_predict, average='micro'))
        print('f1-score: %.4f' % f1_score(self.y_test, self.y_predict, average='micro'))


if __name__ == '__main__':
    # 传入图片路径和需要保存的文件名
    photo_path = './Yale/{}/s{}.bmp'
    save_file = 'yale_data.txt'
    recognition = FaceRecognition(photo_path=photo_path, save_file=save_file)

    recognition.handle_data()
    recognition.load_data()

    acc, model = recognition.train_model()
    print('测试集上的预测准确率为:{}'.format(acc))
    recognition.draw_ROC()
    recognition.model_evaluation()
