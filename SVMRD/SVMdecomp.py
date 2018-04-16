#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: decomp.py      @software: PyCharm 
@time: 2018/4/15 16:55 @site: www.chenymcan.com
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



class SVMdecom(object):
    def __init__(self, X_train, y_train, *kernel):
        self.X_train = X_train
        self.y_train = y_train
        self.n_train = len(X_train)
        self.kernel = kernel

    def get_D(self,X_1, X_2): # n*l, m*l
        n_X_1, n_X_2 = len(X_1), len(X_2)  # n, m
        temp_X_1 = np.array([X_1] * n_X_2) # m*n*l
        temp_X_2 = np.array([X_2] * n_X_1) # n*m*l
        temp_X_1 = np.transpose(temp_X_1, (1, 0, 2)) # n*m*l
        temp = (temp_X_1 - temp_X_2)**2 # n*m*l
        D = np.sum(temp, axis=2) # n*m
        return D


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
        A = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])  # Lagrange multipliers 拉格朗日乘子

    def get_hat_K__(self, K__, K_, alpha_y, mod_w):
        m = len(K__)
        hat_K__ = np.zeros_like(K__)
        for i in range(m):
            for j in range(m):
                hat_K__[i,j] = K__[i,j]-np.dot(alpha_y,K_[:,i])*np.dot(alpha_y,K_[:,j])/mod_w
        return hat_K__

    def get_hat_K_(self, K_,K, alpha_y, mod_w):
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

    def update_D(self, hat_K):
        n = hat_K.shape[0]
        D = np.zeros_like(hat_K)
        for i in range(n):
            for j in range(n):
                D[i,j] = hat_K[i,i]-2*hat_K[i,j]+hat_K[j,j]
        return D

    def update_D_(self,hat_K, hat_K_, hat_K__):
        n,m = hat_K_.shape
        D = np.zeros_like(hat_K_)
        for i in range(n):
            for j in range(m):
                D[i,j] = hat_K[i, i] - 2 * hat_K_[i, j] + hat_K__[j, j]
        return D

    def update_D__(self, hat_K__):
        m = hat_K__.shape[0]
        D = np.zeros_like(hat_K__)
        for i in range(m):
            for j in range(m):
                D[i,j] = hat_K__[i, i] - 2 * hat_K__[i, j] + hat_K__[j, j]
        return D

    # def update_D(self, hat_K):
    #     n,m = hat_K.shape
    #     D = np.zeros_like(hat_K)
    #     for i in range(n):
    #         for j in range(m):
    #             D[i,j] = hat_K[i, i] - 2 * hat_K[i, j] + hat_K[j, j]
    #     return D

    def decom(self,X_test, m = 3):
        i = 0
        n_test =  len(X_test)
        X_train_decom = np.zeros((self.n_train, m))
        X_test_decom = np.zeros((n_test, m))

        D = self.get_D(self.X_train, self.X_train)
        D_ = self.get_D(self.X_train, X_test)
        D__ = self.get_D(X_test, X_test)

        # K = self.kernel[0](D)
        # K_ = self.kernel[0](D_)
        # K__ = self.kernel[0](D__)
        #
        # alpha = self.compute_multipliers(K)
        # alpha_y = np.multiply(alpha, self.y_train)
        # mod_w = np.dot(np.dot(alpha_y, K), alpha_y)
        #
        # X_train_decom[:, 0] = np.dot(alpha_y, K) / mod_w  # 1*n
        # X_test_decom[:, 0] = np.dot(alpha_y, K_) / mod_w  # 1*m
        #
        # hat_K__ = self.get_hat_K__(K__,K_,alpha_y,mod_w)
        # hat_K_ = self.get_hat_K_(K_,K,alpha_y,mod_w)
        # hat_K = self.get_hat_K(K,alpha_y,mod_w)
        #
        # D = self.update_D(hat_K)
        # D_ = self.update_D_(hat_K,hat_K_,hat_K__)
        # D__ = self.update_D__(hat_K__)
        #
        #
        # K = self.kernel[1](D)
        # K_ = self.kernel[1](D_)
        # K__ = self.kernel[1](D__)
        # alpha = self.compute_multipliers(K)
        # alpha_y = np.multiply(alpha, self.y_train)
        # mod_w = np.dot(np.dot(alpha_y, K), alpha_y)
        # X_train_decom[:, 1] = np.dot(alpha_y, K) / mod_w  # 1*n
        # X_test_decom[:, 1] = np.dot(alpha_y, K_) / mod_w  # 1*m
        # hat_K__ = self.get_hat_K__(K__, K_, alpha_y, mod_w)
        # hat_K_ = self.get_hat_K_(K_, K, alpha_y, mod_w)
        # hat_K = self.get_hat_K(K, alpha_y, mod_w)
        # D = self.update_D(hat_K)
        # D_ = self.update_D_(hat_K,hat_K_,hat_K__)
        # D__ = self.update_D__(hat_K__)
        #
        # K = self.kernel[2](D)
        # K_ = self.kernel[2](D_)
        # K__ = self.kernel[2](D__)
        # alpha = self.compute_multipliers(K)
        # alpha_y = np.multiply(alpha, self.y_train)
        # mod_w = np.dot(np.dot(alpha_y, K), alpha_y)
        # X_train_decom[:, 2] = np.dot(alpha_y, K) / mod_w  # 1*n
        # X_test_decom[:, 2] = np.dot(alpha_y, K_) / mod_w  # 1*m
        # hat_K__ = self.get_hat_K__(K__, K_, alpha_y, mod_w)
        # hat_K_ = self.get_hat_K_(K_, K, alpha_y, mod_w)
        # hat_K = self.get_hat_K(K, alpha_y, mod_w)
        # D = self.update_D(hat_K)
        # D_ = self.update_D_(hat_K,hat_K_,hat_K__)
        # D__ = self.update_D__(hat_K__)

        for i in range(m):
            K = self.kernel[i](D)
            K_ = self.kernel[i](D_)
            K__ = self.kernel[i](D__)
            alpha = self.compute_multipliers(K)
            alpha_y = np.multiply(alpha, self.y_train)
            mod_w = np.dot(np.dot(alpha_y, K), alpha_y)
            X_train_decom[:, i] = np.dot(alpha_y, K) / mod_w  # 1*n
            X_test_decom[:, i] = np.dot(alpha_y, K_) / mod_w  # 1*m
            hat_K__ = self.get_hat_K__(K__, K_, alpha_y, mod_w)
            hat_K_ = self.get_hat_K_(K_, K, alpha_y, mod_w)
            hat_K = self.get_hat_K(K, alpha_y, mod_w)
            D = self.update_D(hat_K)
            D_ = self.update_D_(hat_K, hat_K_, hat_K__)
            D__ = self.update_D__(hat_K__)

        return X_train_decom,X_test_decom

    def plot(self, X_train_decom, X_test_decom, y_test, dataname,u=1):
        fig = plt.figure()
        fig.suptitle("对{}数据集降维".format(dataname) ) # +"，高斯核u="+str(u),
        plt.subplot(2, 2, 1)
        X_1, X_0 = X_train_decom[self.y_train == 1], X_train_decom[self.y_train == -1]
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
        X_1, X_0 = X_train_decom[self.y_train == 1], X_train_decom[self.y_train == -1]
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
    # 降维，再knn分类
    gauss_ker_1 = Kernel.gaussian(30)
    gauss_ker_2 = Kernel.gaussian(0.6)
    gauss_ker_3 = Kernel.gaussian(0.6)
    gauss_ker_4 = Kernel.gaussian(10)
    gauss_ker = [gauss_ker_1,gauss_ker_2,gauss_ker_3,gauss_ker_4]
    decomp = SVMdecom(X_train,y_train,*gauss_ker)
    X_train_decom, X_test_decom = decomp.decom(X_test,m=4)

    _1nn = KNeighborsClassifier(1)
    _1nn.fit(X_train,y_train)
    print("1nn before decomp:",_1nn.score(X_test,y_test))

    _1nn.fit(X_train_decom[:,[0,1]],y_train)
    print("1nn after decomp 2:",_1nn.score(X_test_decom[:,[0,1]],y_test))

    _1nn.fit(X_train_decom[:,[0,1,2]],y_train)
    print("1nn after decomp 3:",_1nn.score(X_test_decom[:,[0,1,2]],y_test))

    _1nn.fit(X_train_decom, y_train)
    print("1nn after decomp 4:", _1nn.score(X_test_decom, y_test))

    decomp.plot(X_train_decom,X_test_decom,y_test,"D094")
