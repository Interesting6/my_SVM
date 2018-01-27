#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: py3.5        @license: Apache Licence
@author: 'Treamy'    @contact: chenymcan@gmail.com
@file: svm_train.py      @software: PyCharm
@time: 2018/1/26 13:46 @site: www.ymchen.cn
"""

import my_svm
import kernel
import numpy as np
from sklearn.model_selection import train_test_split

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


def run():
    pass


if __name__ == "__main__":
    x,y = load_data("heart_scale")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=0)
    kernel_ = kernel.Kernel.gaussian(0.5)
    # print(train_y,train_x)

    svm = my_svm.SVM(kernel=kernel_,c=0.431029,)
    svm = svm.training(X_train,y_train)
    accuracy = svm.calc_accuracy(X_train,y_train)
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    accuracy = svm.calc_accuracy(X_test,y_test)
    print("The testing accuracy is: %.3f%%" % (accuracy * 100))

