import numpy as np
import cvxopt.solvers
import logging


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVM(object):
    """docstring for SVM"""
    def __init__(self,kernel,c):
        self._kernel = kernel
        self._c = c

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        # TODO(tulloch) - vectorize
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def training(self, X, y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        self._X_train,self._y_train = X, y
        lagrange_multipliers = self._compute_multipliers(X, y)
        # alpha > 0 时，点位于软间隔内的支持向量
        self.support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        self._support_multipliers = lagrange_multipliers[self.support_vector_indices]
        self._support_vectors = X[self.support_vector_indices]
        self._support_vector_labels = y[self.support_vector_indices]
        self._weights = np.mat(lagrange_multipliers).T
        self._K = np.mat(self._K)

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)  对于软间隔支持向量有这个b=y真-Σalpha*y*K
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        bias = np.mean([y_k - self.predict(x_k,type_="train") for (y_k, x_k) in \
            zip(self._support_vector_labels, self._support_vectors)] )
        self._bias = bias
        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)
        print("svm model training done")
        return self

    def predict(self, x,type_="predict"):
        """
        Computes the SVM prediction on the given features x.
        """
        if type_ == "predict":
            result = self._bias # 在训练时传入的为0故每次均初始为0，预测时（训练好后self._bias为b）为为训练好的b
        else: # "train"
            result = 0.0
        for z_i, x_i, y_i in zip(self._support_multipliers,self._support_vectors,self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i.A[0], x.A[0])
        return np.sign(result).item()


    def _compute_multipliers(self, X, y):
        # 通过cvxopt求拉格朗日乘子
        n_samples, n_features = X.shape

        self._K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * self._K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack))) # 上下合并
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])  # Lagrange multipliers 拉格朗日乘子

    def calc_accuracy(self, test_x, test_y):
        """
        calculates the accuracy.

        """
        n_samples = np.shape(test_x)[0]
        correct = 0.0
        for i in range(n_samples):
            predict = self.predict(test_x[i, :])
            if predict == np.sign(test_y[i]):
                correct += 1
        accuracy = correct / n_samples
        return  accuracy

