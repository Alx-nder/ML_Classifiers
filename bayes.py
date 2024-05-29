import pandas as pd
import numpy as np
from math import log

train100=pd.read_excel('Proj4Train100.xlsx',header=None)
train1000=pd.read_excel('Proj4Train1000.xlsx',header=None)

c1_100=train100.where(train100[5]==1).dropna()[[0,1,2,3,4,]].values
c2_100=train100.where(train100[5]==2).dropna()[[0,1,2,3,4,]].values

c1_1000=train1000.where(train1000[5]==1).dropna()[[0,1,2,3,4,]].values
c2_1000=train1000.where(train1000[5]==2).dropna()[[0,1,2,3,4,]].values

x100=train100[[0,1,2,3,4]].values
x1000=train1000[[0,1,2,3,4]].values
y100=train100[[5]].values
y1000=train1000[[5]].values

test=pd.read_excel('Proj4Test.xlsx',header=None)
X_test=test[[0,1,2,3,4]].values

y_test=test[[5]].values.reshape(-1)

def mlebayes(testsize):
    if testsize==1000:
        def gauss1(x):
            P=len(c1_1000)/len(train1000)
            S=np.cov(c1_1000.T)
            m=(np.ones([1,c1_1000.T.shape[1]]).dot(c1_1000)/c1_1000.shape[0])[0]
            return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))

        def gauss2(x):
            P=len(c2_1000)/len(train1000)
            S=np.cov(c2_1000.T) 
            m=(np.ones([1,c2_1000.T.shape[1]]).dot(c2_1000)/c2_1000.shape[0])[0]
            return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))
    else:
        def gauss1(x):
            P=len(c1_100)/len(train100)
            S=np.cov(c1_100.T)
            m=(np.ones([1,c1_100.T.shape[1]]).dot(c1_100)/c1_100.shape[0])[0]
            return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))

        def gauss2(x):
            P=len(c2_100)/len(train100)
            S=np.cov(c2_100.T) 
            m=(np.ones([1,c2_100.T.shape[1]]).dot(c2_100)/c2_100.shape[0])[0]
            return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))        
    test=pd.read_excel('Proj4Test.xlsx',header=None)
    X_test=test[[0,1,2,3,4]].values
    y_test=test[[5]].values.reshape(-1)
    pred=[1 if gauss1(x)>gauss2(x) else 2 for x in X_test]
    return 1-np.sum(pred==y_test)/len(y_test)

def optimalbayes():
    def gauss1(x):
        P=0.5
        S=[[0.8, 0.2, 0.1, 0.05, 0.01],
        [0.2, 0.7, 0.1, 0.03, 0.02],
        [0.1, 0.1, 0.8, 0.02, 0.01],
        [0.05, 0.03, 0.02, 0.9, 0.01],
        [0.01 ,0.02, 0.01, 0.01, 0.8]]
        S=np.array(S)
        m=[0,0,0,0,0]
        return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))

    def gauss2(x):
        P=0.5
        S=[[0.9, 0.1, 0.05, 0.02, 0.01],
        [0.1, 0.8, 0.1, 0.02, 0.02],
        [0.05, 0.1, 0.7, 0.02, 0.01],
        [0.02, 0.02, 0.02, 0.6, 0.02],
        [0.01, 0.02, 0.01, 0.02, 0.7]] 
        S=np.array(S)    
        m=[1,1,1,1,1]
        return (((-0.5*(x-m).T).dot(np.linalg.inv(S))).dot(x-m)) + np.log(P)+ (-0.5*len(x)*(log(2*np.pi))) -0.5*log(np.linalg.det(S))
    pred=[1 if gauss1(x)>gauss2(x) else 2 for x in X_test]
    return 1-np.sum(pred==y_test)/len(y_test)

class naivebayes:

    def fit(self, X,Y):
        Y=Y.reshape(-1)
        n_samples, n_features = X.shape
        self._classes= np.unique(Y)
        n_classes = len(self._classes)

        # parameters
        self._mean=np.zeros((n_classes,n_features))
        self._var=np.zeros((n_classes,n_features))
        self._priors=np.zeros((n_classes))

        for i, c in enumerate(self._classes):
            X_c=X[Y==c]
            self._mean[i,:]=X_c.mean(axis=0)
            self._var[i,:]=X_c.var(axis=0)
            self._priors[i]=X_c.shape[0] / float(n_samples)


    def predict(self, X):
        y_pred=[self._predict(x) for x in X]
        return np.array(y_pred)
        
    def _predict(self, x):
        posteriors=[]

        for i, c in enumerate(self._classes):
            prior= np.log(self._priors[i])
            posterior=np.sum(np.log(self._pdf(i,x)))
            posterior=posterior+prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self,class_idx,x):
        mean=self._mean[class_idx]
        var=self._var[class_idx]
        top=np.exp(-((x-mean)**2)/(2*var))
        lower=np.sqrt(2*np.pi*var)
        return top/lower
    
if __name__ == "__main__":
    def accuracy(y_true,y_pred):
        acc=np.sum(y_true==y_pred)/len(y_true)
        return 1-acc
    
    nb=naivebayes()
    nb.fit(x100, y100)
    predictions=nb.predict(X_test)
    
    nb2=naivebayes()
    nb2.fit(x1000, y1000)
    predictions2=nb2.predict(X_test)


    print("naive error for 100 samples:",  accuracy(y_test,predictions))
    print("naive error for 1000 samples:",  accuracy(y_test,predictions2))
    print("MLE error for 100 samples:",  mlebayes(100))
    print("MLE error for 100 samples:",  mlebayes(1000))
    print("optimal bayes error for 100 samples:",  optimalbayes())
    print("optimal bayes error for 1000 samples:",  optimalbayes())

    