#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: SVM_RD.py      @software: PyCharm 
@time: 2018/2/2 15:10 @site: www.ymchen.cn
@path: E:/python1/exm/SVMRD/
"""

import numpy as np
import SVM
import kernel
import cvxopt.solvers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


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

def gram_mat_train(X, kernel_, ):
    n_samples, n_features = X.shape
    f = lambda x: np.array(list(map(kernel_, X, [x] * n_samples)))
    K = np.array(list(map(f, X)))
    return K

def gram_mat_test(X_train, x_test, kernel_):
    n_samples = X_train.shape[0]
    f = lambda x: np.array(list(map(kernel_, X_train, [x] * n_samples)))
    K_test = np.array(list(map(f, x_test)))
    return K_test.T

def gram_train_trans(alphas, y, K_train_old):
    temp = np.multiply(alphas, y)
    mod_w = np.dot(np.dot(temp, K_train_old), temp)
    i_samples,j_samples = K_train_old.shape
    K_new = np.zeros_like(K_train_old)
    for i in range(i_samples):
        for j in  range(j_samples):
            K_new[i,j] = K_train_old[i, j] - np.dot(temp, K_train_old[:, i]) * np.dot(temp, K_train_old[:, j]) / mod_w
    return K_new

def gram_test_trans(alphas, y, K_train_old, K_test_old):
    temp = np.multiply(alphas, y)
    mod_w = np.dot(np.dot(temp, K_train_old), temp)
    i_samples,j_samples = K_test_old.shape
    K_new = np.zeros_like(K_test_old)
    for i in range(i_samples):
        for j in  range(j_samples):
            K_new[i,j] = K_test_old[i, j] - np.dot(temp, K_train_old[:, i]) * np.dot(temp, K_test_old[:, j]) / mod_w
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



def svm_decom(X_train,y_train,X_test, kernel_, m = 4):
    i = 0
    n_samples_train,n_samples_test = X_train.shape[0],X_test.shape[0]
    X_train_decom = np.zeros((n_samples_train, m))
    X_test_decom = np.zeros((n_samples_test, m))
    K_train = gram_mat_train(X_train, kernel_, )
    K_test = gram_mat_test(X_train, X_test, kernel_)
    while i < m:
        alphas = compute_multipliers(y_train, K_train)
        temp = np.multiply(alphas,y_train)
        mod_w = np.dot(np.dot(temp, K_train),temp)

        z_train = np.zeros(n_samples_train)
        for index_t in range(n_samples_train):
            mod_tw = np.dot(temp,K_train[:,index_t])
            z_train[index_t] = mod_tw / mod_w
        X_train_decom[:, i] = z_train

        z_test = np.zeros(n_samples_test)
        for index_t in range(n_samples_test):
            mod_tw = np.dot(temp, K_test[:, index_t])
            z_test[index_t] = mod_tw / mod_w
        X_test_decom[:, i] = z_test
        K_test = gram_test_trans(alphas, y_train, K_train, K_test)
        K_train = gram_train_trans(alphas, y_train, K_train)

        i += 1
    return X_train_decom,X_test_decom


from scipy.io import loadmat
def load_data_set(file_path,a=1,b=2):
    m = loadmat(file_path)
    X = m["data"].astype("float64")
    y = m["labels"].astype("float64").flatten()
    index = list(map(lambda x,y:x or y,y==a,y==b))
    X,y = X[index,:],y[index]
    print("the shape of X is: ",X.shape)
    y[y==b] = -1.
    if a!=1. : y[y==a] = 1.
    return X,y

def test(path):
    li = []
    X, y = load_data_set(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    kernel_ = kernel.Kernel.gaussian(0.5)
    svm = SVM.SVM(kernel=kernel_,c=0.5,)
    svm = svm.training(X_train,y_train)
    accuracy1 = svm.calc_accuracy(X_train,y_train)
    accuracy = svm.calc_accuracy(X_test,y_test)
    li.append((accuracy1,accuracy))
    # print("(%.3f%%, %.3f%%)" % (accuracy1 * 100, accuracy * 100))

    for i in range(1,7):
        try:
            X_train_decom,X_test_decom = svm_decom(X_train,y_train,X_test, kernel_,i)
            kernel_ = kernel.Kernel.linear()
            svm = SVM.SVM(kernel=kernel_,c=0.5,)
            svm = svm.training(X_train_decom,y_train)
            accuracy1 = svm.calc_accuracy(X_train_decom,y_train)
            accuracy = svm.calc_accuracy(X_test_decom,y_test)
            li.append((accuracy1, accuracy))
            # print("(%.3f%%, %.3f%%)" % (accuracy1 * 100,accuracy * 100))
        except:
            li.append((0,0))
    return li

if __name__ == "__main__":
    # X, y = load_data_set("F:/__identity/activity/论文/data/D069.mat")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
    # # print(X_train.shape,X_test.shape)
    # kernel_ = kernel.Kernel.gaussian(0.5)
    #
    # 不降维直接高斯核SVM分类
    # # kernel_ = kernel.Kernel.gaussian(0.5)
    # # svm = SVM.SVM(kernel=kernel_,c=0.5,)
    # # svm = svm.training(X_train,y_train)
    # # accuracy1 = svm.calc_accuracy(X_train,y_train)
    # # accuracy = svm.calc_accuracy(X_test,y_test)
    # # print("(%.3f%%, %.3f%%)" % (accuracy1 * 100, accuracy * 100))
    #
    # 降维，再线性核SVM分类
    # X_train_decom,X_test_decom = svm_decom(X_train,y_train,X_test, kernel_,m=1)
    # kernel_ = kernel.Kernel.linear()
    # # kernel_ = kernel.Kernel.gaussian(0.5)
    # svm = SVM.SVM(kernel=kernel_,c=0.5,)
    # svm = svm.training(X_train_decom,y_train)
    # accuracy1 = svm.calc_accuracy(X_train_decom,y_train)
    # # print("The training accuracy is: %.3f%%" % (accuracy1 * 100))
    # accuracy = svm.calc_accuracy(X_test_decom,y_test)
    # # print("The testing accuracy is: %.3f%%" % (accuracy * 100))
    # print("(%.3f%%, %.3f%%)" % (accuracy1 * 100,accuracy * 100))


    li = test("F:/__identity/activity/论文/data/D178.mat")
    for i in li:
        print("(%.3f%%, %.3f%%)" % (i[0] * 100, i[1] * 100))


