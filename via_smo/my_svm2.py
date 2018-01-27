#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: py2.7        @license: Apache Licence
@author: 'Treamy'    @contact: chenymcan@gmail.com
@file: my_svm.py      @software: PyCharm
@time: 2018/1/25 16:31 @site: www.ymchen.cn
"""

import numpy as np
import cPickle as pickle

class SVM:
    def __init__(self,dataSet, labels, C, toler, kernel_option):
        self.train_x = dataSet  # 训练数据集
        self.train_y = labels   # 测试数据集
        self.C = C  # 惩罚参数
        self.toler = toler # 迭代的终止条件之一
        self.n_samples  =np.shape(dataSet)[0]   # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.n_samples, 1))) # 拉格朗日乘子（一个全0的列向量）
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))  # 保存E的缓存
        self.kernel_opt = kernel_option # 选用的核函数及其参数
        self.kernel_mat = calc_kernel(self.train_x, self.kernel_opt)    # 核函数的输出

def calc_kernel(train_x, kernel_option):
    """计算核函数的矩阵
    :param train_x(matrix): 训练样本的特征值
    :param kernel_option(tuple):  核函数的类型以及参数
    :return: kernel_matrix(matrix):  样本的核函数的值
    """
    m = np.shape(train_x)[0]
    kernel_matrix = np.mat(np.zeros((m,m)))
    for i in range(m):
        kernel_matrix[:,i] = calc_kernel_value(train_x, train_x[i,:], kernel_option)
    return kernel_matrix

def calc_kernel_value(train_x, train_x_i, kernel_option):
    """样本之间的核函数值
    :param train_x(matrix): 训练样本
    :param train_x_i(matrix):   第i个训练样本 一个行向量
    :param kernel_option(tuple):   核函数的类型以及参数
    :return: kernel_value(matrix):  样本之间的核函数值
    """
    kernel_type = kernel_option[0]
    m = np.shape(train_x)[0]
    kernel_value = np.mat(np.zeros((m,1)))
    if kernel_type == "rbf":  # 高斯核函数
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(m):
            diff = train_x[i, :] - train_x_i
            kernel_value[i] = np.exp(diff*diff.T/(-2.0*sigma**2))  # 分子为差的2范数的平方
    elif kernel_type == "polynomial":
        p = kernel_option[1]
        for i in range(m):
            kernel_value[i] = (train_x[i, :]*train_x_i.T + 1)**p
    else:
        kernel_value = train_x*train_x_i.T  # 直接一个m*m矩阵×一个m*1的矩阵
    return kernel_value

def SVM_training(train_x, train_y, C=1, toler=0.001, max_iter=500, kernel_option = ("",0)):
    train_x_m,train_y_m = np.mat(train_x),np.mat(train_y)
    train_y_m = train_y_m.T if np.shape(train_y_m)[0] == 1 else train_y_m
    # 1.初始化SVM分类器
    svm = SVM(train_x_m, train_y_m, C, toler, kernel_option)
    # 2.开始训练
    entireSet = True
    alpha_pairs_changed = 0
    iteration = 0

    while iteration<max_iter and (alpha_pairs_changed>0 or entireSet):
        print("\t iteration: ",iteration)
        alpha_pairs_changed = 0

        if entireSet:   # 对所有样本
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:   # 对非边界样本
            bound_samples = []
            for i in range(svm.n_samples):
                if svm.alphas[i,0] > 0 and svm.alphas[i,0] < svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm ,x)
            iteration += 1

        if entireSet:
            entireSet = False
        elif alpha_pairs_changed == 0:
            entireSet = True

    return svm

def cal_error(svm, alpha_k):
    """误差值的计算
    :param svm:
    :param alpha_k(int): 选择出来的变量在alphas里面的index k
    np.multiply(svm.alphas,svm.train_y).T 为一个行向量（αy,αy,αy,αy,...,αy）
    :return: error_k(float): 选出来的变量对应的误差值
    """
    output_k = float(np.multiply(svm.alphas,svm.train_y).T * svm.kernel_mat[:,alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k

def select_second_sample_j(svm,alpha_i,error_i):
    """选择第二个变量
    :param svm:
    :param alpha_i(float): alpha_i
    :param error_i(float): E_i
    :return:
    """
    svm.error_tmp[alpha_i] = [1, error_i] # 用来标记已被优化
    candidate_alpha_list = np.nonzero(svm.error_tmp[:,0].A)[0]  # 因为是列向量，列数[1]都为0，只需记录行数[0]
    max_step,max_step,error_j = 0,0,0

    if len(candidate_alpha_list)>1:
        for alpha_k in candidate_alpha_list:
            if alpha_k == alpha_i:
                continue
            error_k = cal_error(svm, alpha_k)
            if abs(error_k-error_i)>max_step:
                max_step = abs(error_k-error_i)
                alpha_j,error_j = alpha_k,error_k
    else:   # 随机选择
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = np.random.randint(0,svm.n_samples)
        error_j = cal_error(svm, alpha_j)
    return alpha_j, error_j


def update_error_tmp(svm, alpha_k):
    """重新计算误差值，并对其标记为已被优化
    :param svm:
    :param alpha_k: 选择出的变量α
    :return:
    """
    error = cal_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]




def choose_and_update(svm, alpha_i):
    """判断和选择两个alpha进行更新
    :param svm:
    :param alpha_i(int): 选出的第一个变量
    :return:
    """
    error_i = cal_error(svm, alpha_i) # 计算第一个样本的E_i
    if (svm.train_y[alpha_i]*error_i<-svm.toler) and (svm.alphas[alpha_i]<svm.C) \
            or (svm.train_y[alpha_i]*error_i>svm.toler) and (svm.alphas[alpha_i]>0):
        # 1.选择第二个变量
        alpha_j, error_j = select_second_sample_j(svm, alpha_i,error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()
        # 2.计算上下界
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0,svm.alphas[alpha_j]-svm.alphas[alpha_i])
            H = min(svm.C, svm.C+svm.alphas[alpha_j]-svm.alphas[alpha_i])
        else:
            L = max(0,svm.alphas[alpha_j]+svm.alphas[alpha_i]-svm.C)
            H = min(svm.C, svm.alphas[alpha_j]+svm.alphas[alpha_i])
        if L == H:
            return 0
        # 3.计算eta
        eta = svm.kernel_mat[alpha_i, alpha_i] + svm.kernel_mat[alpha_j, alpha_j] - 2.0*svm.kernel_mat[alpha_i,alpha_j]
        if eta <= 0: # 因为这个eta>=0
            return 0
        # 4.更新alpha_j
        svm.alphas[alpha_j] += svm.train_y[alpha_j]*(error_i-error_j)/eta
        # 5.根据范围确实最终的j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # 6.判断是否结束
        if abs(alpha_j_old-svm.alphas[alpha_j])<0.00001:
            update_error_tmp(svm, alpha_j)
            return 0
        # 7.更新alpha_i
        svm.alphas[alpha_i] += svm.train_y[alpha_i]*svm.train_y[alpha_j]*(alpha_j_old-svm.alphas[alpha_j])
        # 8.更新b
        b1 = svm.b - error_i - svm.train_y[alpha_i]*svm.kernel_mat[alpha_i,alpha_i]*(svm.alphas[alpha_i]-alpha_i_old) \
            - svm.train_y[alpha_j]*svm.kernel_mat[alpha_i,alpha_j]*(svm.alphas[alpha_j]-alpha_j_old)
        b2 = svm.b - error_j - svm.train_y[alpha_i]*svm.kernel_mat[alpha_i,alpha_j]*(svm.alphas[alpha_i]-alpha_i_old) \
            - svm.train_y[alpha_j]*svm.kernel_mat[alpha_j,alpha_j]*(svm.alphas[alpha_j]-alpha_j_old)
        if 0<svm.alphas[alpha_i] and svm.alphas[alpha_i]<svm.C:
            svm.b = b1
        elif 0<svm.alphas[alpha_j] and svm.alphas[alpha_j]<svm.C:
            svm.b = b2
        else:
            svm.b = (b1+b2)/2.0
        # 9.更新error
        update_error_tmp(svm, alpha_j)
        update_error_tmp(svm, alpha_i)
        return 1
    else:
        return 0

def svm_predict(svm, test_sample_x):
    kernel_value = calc_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
    predict = np.multiply(svm.train_y, svm.alphas).T*kernel_value + svm.b
    return predict

def get_prediction(test_data, svm):
    '''对样本进行预测
    input:  test_data(mat):测试数据
            svm:SVM模型
    output: prediction(list):预测所属的类别
    '''
    m = np.shape(test_data)[0]
    prediction = []
    for i in range(m):
        predict = svm_predict(svm, test_data[i, :])
        prediction.append(str(np.sign(predict)[0, 0]))
    return prediction

def cal_accuracy(svm, test_x, test_y):
    test_x_m,test_y_m = np.mat(test_x),np.mat(test_y)
    test_y_m = test_y_m.T if np.shape(test_y_m)[0] == 1 else test_y_m
    n_samples = np.shape(test_x_m)[0]
    correct = 0.0
    for i in range(n_samples):
        predict = svm_predict(svm, test_x_m[i, :])
        if np.sign(predict) == np.sign(test_y_m[i]):
            correct += 1
    accuracy = correct / n_samples
    return  accuracy

def save_svm_model(svm_model, model_file):
    with open(model_file, "w") as f:
        pickle.dump(svm_model, f)

def load_svm_model(model_file):
    with open(model_file, "r") as f:
        svm_model = pickle.load(f)
    return svm_model

def save_prediction(result_file, prediction):
    '''保存预测的结果
    input:  result_file(string):结果保存的文件
            prediction(list):预测的结果
    '''
    f = open(result_file, 'w')
    f.write(" ".join(prediction))
    f.close()

def run():
    pass


if __name__ == "__main__":
    print("---------1.load data--------")
    data_set, labels = load_data_libsvm("heart_scale")
    # print data_set
    print("--------- 2.load model ----------")
    C,toler,maxIter = 0.6,0.001,500
    svm_model = my_svm.SVM_training(data_set, labels, C, toler, maxIter, kernel_option = ("rbf",0.431029))
    # print(svm_model.alphas,svm_model.C)
    print("------------ 3、cal accuracy --------------")
    accuracy = my_svm.cal_accuracy(svm_model, data_set, labels)
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
