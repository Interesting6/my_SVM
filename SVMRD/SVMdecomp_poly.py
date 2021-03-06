#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: SVMdecomp_poly.py      @software: PyCharm 
@time: 2018/4/18 21:25 @site: www.chenymcan.com
"""


import numpy as np
import numpy.linalg as la
import cvxopt.solvers

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

    @staticmethod
    def _polykernel(dimension, offset=1):
        return lambda inner: (offset + inner) ** dimension



class SVMdecom(object):
    def __init__(self, X_train, y_train, *kernel):
        self.X_train = X_train
        self.y_train = y_train
        self.n_train = len(X_train)
        self.kernel = kernel


    def get_inner_K(self,X_1,X_2):
        f = lambda x: list(map(lambda s, t: np.inner(s,t), [x] * len(X_1), X_2))
        return np.array(list(map(f,X_1)))

    def get_K(self,inner_K, ker):
        # return ker(inner_K)
        K = np.zeros_like(inner_K)
        n,m = inner_K.shape
        for i in range(n):
            for j in range(m):
                K[i,j] = ker(inner_K[i,j])
        return K

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


    def get_hat_K_(self, K_, K, alpha_y, mod_w):
        n,m = K_.shape
        hat_K_ = np.zeros_like(K_)
        for i in range(n):
            for j in range(m):
                hat_K_[i,j] = K_[i,j]-np.dot(alpha_y,K[:,i])*np.dot(alpha_y,K_[:,j])/mod_w
        return hat_K_

    def get_hat_K(self, K, alpha_y, mod_w):
        n = len(K)
        hat_K = np.zeros_like(K)
        for i in range(n):
            for j in range(n):
                hat_K[i,j] = K[i,j]-np.dot(alpha_y,K[:,i])*np.dot(alpha_y,K[:,j])/mod_w
        return hat_K


    def decom(self,X_test, m = 3):
        i = 0
        n_test =  len(X_test)
        X_train_decom = np.zeros((self.n_train, m))
        X_test_decom = np.zeros((n_test, m))
        inner_K = self.get_inner_K(self.X_train, self.X_train)
        inner_K_ = self.get_inner_K(self.X_train, X_test)

        for i in range(m):
            K = self.get_K(inner_K, self.kernel[i])
            K_ = self.get_K(inner_K_, self.kernel[i])
            alpha = self.compute_multipliers(K)
            alpha_y = np.multiply(alpha, self.y_train)
            mod_w = np.dot(np.dot(alpha_y, K), alpha_y)
            X_train_decom[:, i] = np.dot(alpha_y, K) / np.sqrt(mod_w)  # 1*n
            X_test_decom[:, i] = np.dot(alpha_y, K_) / np.sqrt(mod_w)   # 1*m
            inner_K = self.get_hat_K(K,alpha_y,mod_w)
            inner_K_ = self.get_hat_K_(K_,K,alpha_y,mod_w)

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
    dataname = "D094"
    X, y = load_data_set("F:/__identity/activity/论文/data/{}.mat".format(dataname))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # 降维，再knn分类
    poly_ker_1 = Kernel._polykernel(2)
    poly_ker_2 = Kernel._polykernel(2)
    poly_ker_3 = Kernel._polykernel(2)
    poly_ker_4 = Kernel._polykernel(2)

    poly_ker = [poly_ker_1,poly_ker_2,poly_ker_3,poly_ker_4]
    decomp = SVMdecom(X_train,y_train,*poly_ker)
    m = 4
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


    SVMdecom.plot(X_train_decom,y_train, X_test_decom,y_test,dataname)
