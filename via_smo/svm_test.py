#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: svm_train.py      @software: PyCharm 
@time: 2018/1/26 13:46 @site: www.ymchen.cn
"""

import my_svm
import numpy as np

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
    train_x,train_y = load_data("heart_scale")
    # print(train_y,train_x)
    svm = my_svm.SVM(C=0.6,kernel_option=("rbf",0.431029))
    svm = svm.SVM_training(train_x,train_y,)
    # print(svm.alphas,svm.b)
    accuracy = svm.get_train_accracy()
    print("The training accuracy is: %.3f%%" % (accuracy * 100))


