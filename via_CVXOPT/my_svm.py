import numpy as np
import kernel
import cvxopt.solvers
import matplotlib.pyplot as plt


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVM(object):
    """docstring for SVM"""
    def __init__(self,kernel,c=1):
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
        self._X_train,self._y_train = (X.A, y.A.flatten()) if type(X)==np.matrixlib.defmatrix.matrix else (X,y)

        lagrange_multipliers = self._compute_multipliers(X, y)
        # alpha > 0 时，点位于软间隔内的支持向量
        self._support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        self._support_vectors_num = len(self._support_vector_indices)
        # 统一为数组形式，使得行（或列）向量为一维数组，X为二维数组
        self._support_multipliers = lagrange_multipliers[self._support_vector_indices]
        self._support_vectors = self._X_train[self._support_vector_indices]
        self._support_vector_labels = self._y_train[self._support_vector_indices]
        # self._weights = lagrange_multipliers

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)  对于软间隔支持向量有这个b=y真-Σalpha*y*K
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        bias = np.mean([y_k - self.predict(x_k,type_="train") for (y_k, x_k) in \
            zip(self._support_vector_labels, self._support_vectors)] )
        self._bias = bias
        # logging.info("Bias: %s", self._bias)
        # logging.info("Weights: %s", self._weights)
        # logging.info("Support vectors: %s", self._support_vectors)
        # logging.info("Support vector labels: %s", self._support_vector_labels)
        print("svm model training done")
        return self


    def predict(self, x,type_="predict"):
        """
        Computes the SVM prediction on the given features x.
        """
        if type_ == "predict":
            result = self._bias # 在预测时传入训练好后self._bias（训练好的b）
            tmp = np.multiply(self._support_multipliers, self._support_vector_labels)
            tmp2 = np.array(list(map(self._kernel, self._support_vectors,[x]*self._support_vectors.shape[0])))
            result += np.dot(tmp,tmp2)
        else: # "train"
            result = 0.0 # 在训练时传入的为0，故每次均初始为0
            vector_indices_x = self._X_train.tolist().index(x.tolist())
            result += np.dot(np.multiply(self._support_multipliers,self._support_vector_labels),
                             self._K[self._support_vector_indices, vector_indices_x])
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

    def get_function(self):
        if self._kernel == kernel.Kernel.linear():
            tmp = map(lambda x,y,z:x*y*z, self._support_multipliers, self._support_vector_labels,self._support_vectors)
            self.w = sum(tmp)
            b = self._bias
            return lambda x:np.dot(self.w,x)+b
        elif "gaussian" in str(self._kernel):
            kernel_ = self._kernel
            tmp = np.array(list(map(lambda a, b: a * b , self._support_multipliers, self._support_vector_labels,)))
            return lambda x:np.dot(tmp,np.array(list(
                map(kernel_,self._support_vectors,[x]*self._support_vectors_num)
            )))+ self._bias
        else:
            return 0


    def predict_data_set(self,x):
        if self._kernel == kernel.Kernel.linear():
            tmp = map(lambda x,y,z:x*y*z, self._support_multipliers, self._support_vector_labels,self._support_vectors)
            self.w = sum(tmp)
            b = self._bias
            # return lambda x:np.dot(self.w,x)+b
            predict_f = lambda x: np.dot(self.w, x) + b
            return np.array(list(map(predict_f, x)))
        elif "gaussian" in str(self._kernel):
            tmp_li = []
            for x_i in x:
                result = self._bias
                kernel_ = self._kernel
                for a_i,y_i,sv_i in zip(self._support_multipliers,self._support_vector_labels,
                        self._support_vectors):
                    result += a_i*y_i*kernel_(sv_i,x_i)
                tmp_li.append(result)
            return np.array(tmp_li)
        else:
            return 0


    def show_data_set(self, X, y,):
        # 作training sample数据点的图
        X,y = (X.A, y.A.flatten()) if type(X)==np.matrixlib.defmatrix.matrix else (X,y)
        x1_min, x1_max = np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5
        x2_min, x2_max = np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5
        X_1, X_0 = X[y == 1], X[y == -1]
        plt.plot(X_1[:, 0], X_1[:, 1], "ro")
        plt.plot(X_0[:, 0], X_0[:, 1], "bo")
        # 做support vectors 的图
        try:
            if (X == self._X_train).all():
                plt.scatter(self._support_vectors[:, 0], self._support_vectors[:, 1], s=100, c="g")
                plt.title("SVM classifier in training data set")
        except:
            plt.title("SVM classifier in testing data set")
        # pl.contour做等值线图
        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50))
        X_ = np.array([[x1, x2] for x1, x2 in zip(X1.flatten(), X2.flatten())])  # x=(x1,x2)，得到所有的(x1,x2)即平面上的点
        Y = self.predict_data_set(X_).reshape(X1.shape)

        plt.contour(X1, X2, Y, [0.0], colors="k", linewidths=1, origin='lower')
        # 第四个参数只要Y为0的时候的那根线，故predict里面不要sign，否则下面俩都为0
        plt.contour(X1, X2, Y - 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X1, X2, Y + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.xticks(())
        plt.yticks(())
        plt.axis("tight")
        plt.show()

