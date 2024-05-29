import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
import matplotlib.pyplot as plt

def gaussian_kernel(x, y, sigma=1.75):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=gaussian_kernel, C=102.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n_samples= X.shape[0]

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        G = cvxopt.matrix(np.vstack((np.eye(n_samples)*-1,np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = cvxopt.matrix(y.reshape(1,-1)*1.0)
        b = cvxopt.matrix(0.0)

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lambdas = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = lambdas > 0.09999
        ind = np.arange(len(lambdas))[sv]
        self.lambdas = lambdas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Offset
        self.b = 0
        for n in range(len(self.lambdas)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.lambdas * self.sv_y * K[ind[n],sv])
        self.b /= len(self.lambdas)
        
        # Weight vector
        self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for lambdas, sv_y, sv in zip(self.lambdas, self.sv_y, self.sv):
                    s += lambdas * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == '__main__':
        def get_data():
            df=pd.read_excel('Proj2DataSet.xlsx',header=None)
            df=df.set_axis(['x1', 'x2', 'class'], axis='columns')
            c1=df.where(df['class']==1).dropna()[['x1','x2']].values
            c2=df.where(df['class']==-1).dropna()[['x1','x2']].values
            y=df[['class']].values
            return c1, y[:len(c1)], c2, y[len(c1):]

        def split_train(X1, y1, X2, y2):
            X_train = np.vstack((X1, X2))
            y_train = np.hstack((y1.flatten(), y2.flatten()))
            return X_train, y_train

        def split_test(X1, y1, X2, y2):
            X_test = np.vstack((X1, X2))
            y_test = np.hstack((y1.flatten(), y2.flatten()))
            return X_test, y_test

        
        def plot_contour(X1_train, X2_train, clf):
            
            plt.scatter(X1_train[:,0], X1_train[:,1], c='green',label='class 1')
            plt.scatter(X2_train[:,0], X2_train[:,1], c='orange', label='class 2')
            
            # support vectors
            plt.scatter(clf.sv[:,0], clf.sv[:,1], s=10, facecolors='none', edgecolors='r',label='support vectors')
            
            X1, X2 = np.meshgrid(np.linspace(-3,8,50), np.linspace(-3,9,50))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
            Z = clf.project(X).reshape(X1.shape)
            plt.contour(X1, X2, Z, [0.0], colors='darkblue', labels='decision boundary')
            plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linestyles='--')
            plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linestyles='--')

            plt.annotate(f'No. Support vectors:{len(clf.sv)} \nC:{clf.C}', xy=(6,4), xycoords='data',
                xytext=(-100,60), textcoords='offset points')
            
            plt.title("SVM")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend(loc="lower left")
            plt.show()

        def rbf_svm(C):
            X1, y1, X2, y2 = get_data()
            X_train, y_train = split_train(X1, y1, X2, y2)
            X_test, y_test = split_test(X1, y1, X2, y2)

            clf = SVM(gaussian_kernel,C)
            clf.fit(X_train, y_train)

            y_predict = clf.predict(X_test)
            correct = np.sum(y_predict == y_test)

            plt.annotate(f'No. misclassified samples:{len(y_predict)-correct}', xy=(6,5), xycoords='data',
                xytext=(-100,60), textcoords='offset points')
            
            plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
            
        # rbf_svm(100)
        rbf_svm(10)