import pandas as pd
import numpy as np
from math import log
import preprocessing

Xtrain=preprocessing.kag_X_train
Xtest=preprocessing.kag_X_test
ytrain=preprocessing.kag_y_train
ytest=preprocessing.kag_y_test

class1=Xtrain[ytrain==1]
class2=Xtrain[ytrain==0]

def mlebayes():
    def gauss1(x):
        P=len(class1)/len(Xtrain)
        S=np.cov(class1.T)
        m=(np.ones([1,class1.T.shape[1]]).dot(class1)/class1.shape[0])[0]
        return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))

    def gauss2(x):
        P=len(class2)/len(Xtrain)
        S=np.cov(class2.T) 
        m=(np.ones([1,class2.T.shape[1]]).dot(class2)/class2.shape[0])[0]
        return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))
    
    pred=[1 if gauss1(x)>gauss2(x) else 2 for x in Xtest]
    return 1-np.sum(pred==ytest)/len(ytest)

print("MLE error:",  mlebayes())

    