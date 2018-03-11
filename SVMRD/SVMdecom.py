
import numpy as np
import cvxopt.solvers
import SVM, kernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.neighbors import KNeighborsClassifier

class my_knn(object):
    """docstring for my_knn"""
    def __init__(self, k, svmdecomp):
        super(my_knn, self).__init__()
        self.k = k
        self.svmdecomp = svmdecomp
        # self.X_test, self.y_test = np.array(X_test), np.array(y_test) X_test, y_test,

    def train(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_test,y_test or y_train was not equail!"
                             "The length of X_test,y_test is %s"
                             "But the length of y_train is %s" % (len(self.X_train), len(self.y_train)))
        return self

    def predict_one(self, X, m=2):
        dist2xtrain = X - self.X_train
        K = self.svmdecomp.gram_mat_train(dist2xtrain,self.svmdecomp.kernel_)
        alphas = self.svmdecomp.alphas
        for i in range(m):
            y_train = self.svmdecomp.y_train
            K = self.svmdecomp.gram_train_trans(alphas,y_train, K)
        dist2xtrain = K.diagonal()**0.5
        # dist2xtrain = np.sum((X - self.X_train)**2, axis=1)**0.5
        index = dist2xtrain.argsort() # 从小到大（近到远）
        label_count = {}
        for i in range(self.k):
            label = self.y_train[index[i]]
            label_count[label] = label_count.get(label, 0) + 1
        # 将label_count的值从大到小排列label_count的键
        y_predict = sorted(label_count, key=lambda x: label_count[x], reverse=True)[0]
        return y_predict

    def predict_all(self, X):
        return np.array(list(map(self.predict_one, X)))

    def calc_accuracy(self, X, y):
        predict = self.predict_all(X)
        total = X.shape[0]
        right = sum(predict == y)
        accuracy = right/total
        return accuracy


class SVMdecom(object):
    def __init__(self, X_train, y_train, kernel_,):
        self.X_train = X_train
        self.y_train = y_train
        self.kernel_ = kernel_

    def gram_mat_train(self, X_train, kernel_, ):
        n_samples = X_train.shape[0]
        f = lambda x: np.array(list(map(kernel_, X_train, [x] * n_samples)))
        K_train = np.array(list(map(f, X_train)))
        return K_train

    def gram_mat_test(self, X_train, X_test, kernel_):
        n_samples = X_train.shape[0]
        f = lambda x: np.array(list(map(kernel_, X_train, [x] * n_samples)))
        K_test = np.array(list(map(f, X_test)))
        return K_test.T

    def gram_train_trans(self, alphas, y, K_train_old):
        temp = np.multiply(alphas, y)
        mod_w = np.dot(np.dot(temp, K_train_old), temp)
        i_samples,j_samples = K_train_old.shape
        K_new = np.zeros_like(K_train_old)
        for i in range(i_samples):
            for j in  range(j_samples):
                K_new[i,j] = K_train_old[i, j] - np.dot(temp, K_train_old[:, i]) * np.dot(temp, K_train_old[:, j]) / mod_w
        return K_new

    def gram_test_trans(self, alphas, y, K_train_old, K_test_old):
        temp = np.multiply(alphas, y)
        mod_w = np.dot(np.dot(temp, K_train_old), temp)
        i_samples,j_samples = K_test_old.shape
        K_new = np.zeros_like(K_test_old)
        for i in range(i_samples):
            for j in  range(j_samples):
                K_new[i,j] = K_test_old[i, j] - np.dot(temp, K_train_old[:, i]) * np.dot(temp, K_test_old[:, j]) / mod_w
        return K_new

    def compute_multipliers(self, y, K, c=1):
        # 通过cvxopt求拉格朗日乘子
        n_samples = y.shape[0]
        # min 1/2 x^T P x + q^T x  s.t.  Gx \coneleq h  &  Ax = b
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack))) # 上下合并
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])  # Lagrange multipliers 拉格朗日乘子

    def decom(self,X_test, m = 2):
        i = 0
        n_samples_train,n_samples_test = self.X_train.shape[0], X_test.shape[0]
        X_train_decom = np.zeros((n_samples_train, m))
        X_test_decom = np.zeros((n_samples_test, m))
        K_train = self.gram_mat_train(self.X_train, self.kernel_, )
        K_test = self.gram_mat_test(self.X_train, X_test, self.kernel_)
        while i < m:
            self.alphas = self.compute_multipliers(self.y_train, K_train)
            temp = np.multiply(self.alphas,self.y_train)
            mod_w = np.dot(np.dot(temp, K_train),temp)

            X_train_decom[:, i] = np.dot(temp, K_train) / mod_w
            X_test_decom[:, i] = np.dot(temp, K_test) / mod_w

            K_test = self.gram_test_trans(self.alphas, self.y_train, K_train, K_test)
            K_train = self.gram_train_trans(self.alphas, self.y_train, K_train)
            i += 1
        self.K_train, self.K_test = K_train, K_test
        return X_train_decom,X_test_decom

    def plot(self, X_train_decom, X_test_decom, y_test, dataname):
        plt.subplot(2, 2, 1)
        X_1, X_0 = X_train_decom[self.y_train == 1], X_train_decom[self.y_train == -1]
        print(len(X_1),len(X_0),"------------")
        plt.scatter(X_1[:, 0], X_1[:, 1], color="r")
        plt.scatter(X_0[:, 0], X_0[:, 1], color="b")
        plt.title("对{}集训练数据降维".format(dataname))

        plt.subplot(2, 2, 2)
        X_1, X_0 = X_test_decom[y_test == 1], X_test_decom[y_test == -1]
        print(len(X_1), len(X_0), "------------")
        plt.scatter(X_1[:, 0], X_1[:, 1], color="r")
        plt.scatter(X_0[:, 0], X_0[:, 1], color="b")
        plt.title("对{}集测试数据降维".format(dataname))

        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(223, projection='3d')
        X_1, X_0 = X_train_decom[self.y_train == 1], X_train_decom[self.y_train == -1]
        ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], color="r")
        ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], color="b")
        plt.title("对{}集训练数据降维".format(dataname))
        ax = plt.subplot(224, projection='3d')
        X_1, X_0 = X_test_decom[y_test == 1], X_test_decom[y_test == -1]
        ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], color="r")
        ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], color="b")
        plt.title("对{}集测试数据降维".format(dataname))
        plt.show()

from scipy.io import loadmat
def load_data_set(file_path, a=1, b=2):
    m = loadmat(file_path)
    X = m["data"].astype("float64")
    y = m["labels"].astype("float64").flatten()
    index = list(map(lambda x, y: x or y, y == a, y == b))
    X, y = X[index, :], y[index]
    print("the shape of X is: ", X.shape)
    y[y == b] = -1.
    if a != 1.: y[y == a] = 1.
    return X, y


if __name__ == "__main__":
    dataname = "D199"
    X, y = load_data_set("F:/__identity/activity/论文/data/{}.mat".format(dataname))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
    # 降维，再可视化
    kernel_ = kernel.Kernel.gaussian(0.5)
    svm_decom = SVMdecom(X_train,y_train, kernel_)
    X_train_decom,X_test_decom = svm_decom.decom(X_test,m=3)
    # print(X_train_decom.shape, X_test_decom.shape)

    svm_decom.plot(X_train_decom, X_test_decom, y_test,dataname)

