#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: SVM2.py      @software: PyCharm 
@time: 2018/2/2 15:10 @site: www.ymchen.cn
"""

import numpy as np
import kernel
import cvxopt.solvers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


def gram_matrix(kernel_, X):
    n_samples, n_features = X.shape
    f = lambda x: np.array(list(map(kernel_, X, [x] * n_samples)))
    K = np.array(list(map(f, X)))
    return K

def gram_mat_trans(alphas,y,K_old):
    temp = np.multiply(alphas, y)
    mod_w = np.dot(np.dot(temp, K_old), temp)
    n_samples = y.shape[0]
    K_new = np.zeros_like(K_old)
    for i in range(n_samples):
        for j in  range(n_samples):
            K_new[i,j] = K_old[i,j] - np.dot(temp,K_old[:,i])*np.dot(temp,K_old[:,j])/mod_w
    return K_new

def compute_multipliers(y, K, c=1):
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




def load_data(data_file):
    data_set, labels = [], []
    with open(data_file,"r") as f:
        textlist = f.readlines()
        for line in textlist:
            tmp = []
            line = line.strip().split(" ")
            labels.append(float(line[0]))
            i = 1
            for word in line[1:]:
                feature,value = word.split(":")
                while int(feature) != i:
                    tmp.append(float(0))
                    i += 1
                tmp.append(float(value))
                i += 1
            data_set.append(tmp)

    return (np.mat(data_set),np.mat(labels).T)




if __name__ == "__main__":
    X,y = load_data("heart_scale")
    X,y = (X.A, y.A.flatten()) if type(X) == np.matrixlib.defmatrix.matrix else (X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)

    kernel_ = kernel.Kernel.gaussian(0.5)
    K = gram_matrix(kernel_, X_train)
    i = 0
    m = 5
    n_samples = X_train.shape[0]
    X_train_decom = np.zeros((n_samples,m))
    while i < m:
        alphas = compute_multipliers(y_train, K)
        temp = np.multiply(alphas,y_train)
        mod_w = np.dot(np.dot(temp, K),temp)
        z = np.zeros(n_samples)
        for index_t in range(n_samples):
            t = X_train[index_t,:]
            mod_tw = np.dot(temp,K[:,index_t])
            z[index_t] = mod_tw / mod_w
        X_train_decom[:, i] = z
        K = gram_mat_trans(alphas,y_train, K)
        i += 1
    print(X_train_decom[:5,:])
    import SVM
    svm = SVM.SVM(kernel=kernel_,c=0.5,)
    svm = svm.training(X_train_decom,y_train)
    accuracy = svm.calc_accuracy(X_train_decom,y_train)
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    # # accuracy = svm.calc_accuracy(X_test,y_test)
    # # print("The testing accuracy is: %.3f%%" % (accuracy * 100))




