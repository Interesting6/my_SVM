#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: test3.py      @software: PyCharm 
@time: 2018/4/27 20:30 @site: www.chenymcan.com
"""



import numpy as np
import numpy.linalg as la
import cvxopt.solvers

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale

class Kernel(object):
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gauss(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def gaussian(sigma):
        return lambda d: np.exp(- d / (2 * sigma ** 2))



class SVMdecom(object):
    def __init__(self, X_train, y_train, *kernel):
        self.X_train = X_train
        self.y_train = y_train
        self.n_train = len(X_train)
        self.kernel = kernel

    def get_D(self,X_1,X_2):
        f = lambda x: list(map(lambda s, t: sum((s - t) ** 2), [x] * len(X_1), X_2))
        return np.array(list(map(f,X_1)))

    def compute_multipliers(self, K, c=1):
        # 通过cvxopt求拉格朗日乘子
        n = self.n_train
        y = self.y_train
        # min 1/2 x^T P x + q^T x  s.t.  Gx \coneleq h  &  Ax = b
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n))
        G_std = cvxopt.matrix(np.diag(np.ones(n) * -1))
        h_std = cvxopt.matrix(np.zeros(n))
        G_slack = cvxopt.matrix(np.diag(np.ones(n)))
        h_slack = cvxopt.matrix(np.ones(n) * c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack))) # 上下合并
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n),"d")
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])  # Lagrange multipliers 拉格朗日乘子

    def update_G(self, K, Z ):
        G = K - np.outer(Z,Z)
        return G

    def update_D(self, G):  # update D & D__
        diag = np.diag(G)
        D = diag.reshape((len(diag), 1)) - 2 * G + diag
        return D

    def decom(self,X_test, m = 3):
        n_test =  len(X_test)

        X = np.vstack((X_train,X_test))
        X_decom = np.zeros((len(X), m))

        # G = np.inner(X,X)
        D = self.get_D(X, X)
        for i in range(m):
            K = self.kernel[i](D)
            alpha = self.compute_multipliers(K[:self.n_train,:self.n_train])
            alpha_y = np.multiply(alpha, self.y_train)
            mod_w = np.dot(np.dot(alpha_y, K[:self.n_train,:self.n_train]), alpha_y)
            Z = np.dot(alpha_y, K[:self.n_train,:]) / np.sqrt(mod_w)  # 1*(n+m)
            X_decom[:, i] = Z
            G = self.update_G(K, Z)
            D = self.update_D(G)
        X_train_decom,X_test_decom = X_decom[:self.n_train,:],X_decom[self.n_train:,:]
        return X_train_decom,X_test_decom

    @staticmethod
    def plot(X_train_decom,y_train, X_test_decom, y_test, dataname,u=1):
        fig = plt.figure()
        fig.suptitle("对{}数据集降维".format(dataname) ) # +"，高斯核u="+str(u),
        plt.subplot(2, 2, 1)
        X_1, X_0 = X_train_decom[y_train == 1], X_train_decom[y_train == -1]
        plt.scatter(X_1[:, 0], X_1[:, 1], color="r", marker=".")
        plt.scatter(X_0[:, 0], X_0[:, 1], color="b", marker=".")
        plt.title("训练数据2D图")

        plt.subplot(2, 2, 2)
        X_1, X_0 = X_test_decom[y_test == 1], X_test_decom[y_test == -1]
        plt.scatter(X_1[:, 0], X_1[:, 1], color="r", marker=".")
        plt.scatter(X_0[:, 0], X_0[:, 1], color="b", marker=".")
        plt.title("测试数据2D图")

        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(223, projection='3d')
        X_1, X_0 = X_train_decom[y_train == 1], X_train_decom[y_train == -1]
        ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], color="r")
        ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], color="b")
        plt.title("训练数据3D图")
        ax = plt.subplot(224, projection='3d')
        X_1, X_0 = X_test_decom[y_test == 1], X_test_decom[y_test == -1]
        ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], color="r")
        ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], color="b")
        plt.title("测试数据2D图")
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
    dataname = "D095"
    X, y = load_data_set("F:/__identity/activity/论文/data/{}.mat".format(dataname))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    X = minmax_scale(X)
    X_train,y_train = X[:500],y[:500]
    X_test,y_test = X[500:],y[500:]


    # 降维，再knn分类
    # sigma = [1, 1.2, 0.3, 0.1,  0.01,1e-2,1e-3,1e-4,1e-5]
    sigma = [17.3376, 7.2154, 1.1586, 4.9305, 6.2592]
    gauss_ker = [Kernel.gaussian(i) for i in sigma]
    decomp = SVMdecom(X_train,y_train,*gauss_ker)
    m = 5
    X_train_decom, X_test_decom = decomp.decom(X_test,m=m)


    _1nn = KNeighborsClassifier(1)
    _1nn.fit(X_train,y_train)
    print("1nn before decomp:",_1nn.score(X_test,y_test))

    for i in range(m):
        _1nn.fit(X_train_decom[:,range(i+1)],y_train)
        print("1nn after SVM decomp {}:".format(str(i+1)), _1nn.score(X_test_decom[:,range(i+1)],y_test))

    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train,y_train)
    print("svm in origin",svm.score(X_test,y_test))

    svm.fit(X_train_decom, y_train)
    print("svm in decomp", svm.score(X_test_decom, y_test))
    for i in range(m):
        svm.fit(X_train_decom[:,range(i+1)],y_train)
        print("SVM after SVM decomp {}:".format(str(i+1)), svm.score(X_test_decom[:,range(i+1)],y_test))


    SVMdecom.plot(X_train_decom,y_train, X_test_decom,y_test,dataname)
