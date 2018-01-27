#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: py3.5        @license: Apache Licence
@author: 'Treamy'    @contact: chenymcan@gmail.com
@file: svm_train.py      @software: PyCharm
@time: 2018/1/26 13:46 @site: www.ymchen.cn
"""

import svm_,kernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)
    return X_train, y_train,X_test, y_test

def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2):
    X1_test = X1[90:]
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 70)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 70)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 60)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 60)))
    y2 = np.ones(len(X2)) * -1
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)
    return X_train, y_train, X_test, y_test


def run():
    pass


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = gen_non_lin_separable_data()

    kernel_ = kernel.Kernel.gaussian(5.0)
    svm = svm_.SVM(kernel=kernel_,c=1)
    svm = svm.training(X_train,y_train)
    print("the support vector number is: %d" % svm._support_vectors_num)
    accuracy = svm.calc_accuracy(X_train,y_train)
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    svm.show_data_set(X_train, y_train)
    accuracy = svm.calc_accuracy(X_test,y_test)
    print("The testing accuracy is: %.3f%%" % (accuracy * 100))
    svm.show_data_set(X_test,y_test)
