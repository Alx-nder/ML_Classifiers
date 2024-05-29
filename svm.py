import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def p1SVM(C):
    df=pd.read_excel('proj2dataset.xlsx',header=None)
    # rename columns
    df=df.set_axis(['x1', 'x2', 'class'], axis='columns')

    c1=df.where(df['class']==1).dropna()[['x1','x2']].values
    c2=df.where(df['class']==-1).dropna()[['x1','x2']].values

    # feature vectors
    X=df[['x1','x2']].values
    # class labels
    y=df[['class']].values

    m,n = X.shape

    Xy = y * X

    # row col dot product
    H = np.dot(Xy , Xy.T)

    # prepare for form 1/2(x'Px+q'x) s.t. Gx<=h; Ax=b
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.T*1.0)
    b = cvxopt_matrix(np.zeros(1))
    # solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(sol['x'])

    # weight vector
    w = np.matmul((y * lambdas).T,X).reshape(-1,1)

    # threshold for lambda is 0.09999
    Sv = ((lambdas>0.09999) ).flatten()
    w0 = np.mean(y[Sv] - np.dot(X[Sv], w))

    misclassed1=(np.dot(c1,w)+w0<0)
    misclassed2=(np.dot(c2,w)+w0>=0)
    misclassed=np.concatenate([misclassed1,misclassed2])

    plt.scatter(x=c1.T[0],y=c1.T[1],c='green',label='class 1')
    plt.scatter(x=c2.T[0],y=c2.T[1],c='orange',label='class 2')

    x1 = np.linspace(-2,8)
    plt.plot(x1, -(x1*w[0] +w0)/w[1], color = 'darkblue',label='decision boundary')
    plt.plot(x1, -(x1*w[0] +w0 -1)/w[1], color = 'grey',linestyle='--')
    plt.plot(x1, -(x1*w[0] +w0 +1)/w[1], color = 'grey',linestyle='--')
    plt.scatter(X[Sv].T[0], X[Sv].T[1], s=10, facecolors='none', edgecolors='r',label='support vectors')


    plt.annotate(f'No. Support vectors:{len(X[Sv])} \nNo. Misclassified samples:{len(y[misclassed])}\nC:{C}', xy=(2,4), xycoords='data',
                xytext=(-100,60), textcoords='offset points')

    plt.title("SVM")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="lower right")
    plt.show()

def compareSVM():
    np.random.seed(0)
    my_sizeNtimes=[]
    sK_sizeNtimes=[]
    d1=np.random.rand(100,2)
    d2=np.random.rand(200,2)
    d3=np.random.rand(300,2)
    d4=np.random.rand(400,2)
    d5=np.random.rand(500,2)
    d6=np.random.rand(600,2)
    d7=np.random.rand(700,2)
    d8=np.random.rand(800,2)
    c1=np.random.choice([-1,1],d1.shape[0]).reshape(-1,1)
    c2=np.random.choice([-1,1],d2.shape[0]).reshape(-1,1)
    c3=np.random.choice([-1,1],d3.shape[0]).reshape(-1,1)
    c4=np.random.choice([-1,1],d4.shape[0]).reshape(-1,1)
    c5=np.random.choice([-1,1],d5.shape[0]).reshape(-1,1)
    c6=np.random.choice([-1,1],d6.shape[0]).reshape(-1,1)
    c7=np.random.choice([-1,1],d7.shape[0]).reshape(-1,1)
    c8=np.random.choice([-1,1],d8.shape[0]).reshape(-1,1)

    # hard-coded SVM
    def mySVM(X,y):    
        C = 100
        m,n = X.shape
        Xy = y * X
        H = np.dot(Xy , Xy.T)

        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = cvxopt_matrix(y.T*1.0)
        b = cvxopt_matrix(np.zeros(1))
        
        start=time.time()
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        end=time.time()

        lambdas = np.array(sol['x'])
        w = np.matmul((y * lambdas).T,X).reshape(-1,1)
        Sv = ((lambdas>0.09999) ).flatten()
        w0 = np.mean(y[Sv] - np.dot(X[Sv], w))
        my_sizeNtimes.append([X.shape[0],(end-start)*10**3])
        return((end-start)*10**3, w, w0 )

    #sklearn SVM
    def skSvm(X,y):
        starts = time.time()
        model = SVC(C = 100, kernel = 'linear')
        model.fit(X, y.reshape(-1)) 
        ends = time.time()
        sK_sizeNtimes.append([X.shape[0],(ends-starts)*10**3])

        return((ends-starts)*10**3 ,model.coef_, model.intercept_)
    mySVM(d1,c1)
    skSvm(d1,c1)

    mySVM(d2,c2)
    skSvm(d2,c2)

    mySVM(d3,c3)
    skSvm(d3,c3)

    mySVM(d4,c4)
    skSvm(d4,c4)

    mySVM(d5,c5)
    skSvm(d5,c5)

    mySVM(d6,c6)
    skSvm(d6,c6)

    mySVM(d7,c7)
    skSvm(d7,c7)

    mySVM(d8,c8)
    skSvm(d8,c8)

    sK_sizeNtimes=np.array(sK_sizeNtimes)
    my_sizeNtimes=np.array(my_sizeNtimes)

    plt.plot(sK_sizeNtimes.T[0], sK_sizeNtimes.T[1],marker='o',label='SVO_SVM')
    plt.plot(my_sizeNtimes.T[0], my_sizeNtimes.T[1],marker='o',label='my_SVM')

    plt.ylim(0,300)
    plt.xlim(100,800)

    plt.xlabel('No. Samples')
    plt.ylabel('Time (ms)')

    plt.legend(loc="upper left")
    plt.title('Time complexity comparison of SVM methods')
    plt.show()


# p1SVM(0.100)
# p1SVM(100)
compareSVM()
