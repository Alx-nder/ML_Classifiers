import numpy as np
import pandas as pd

train100=pd.read_excel('Proj4Train100.xlsx',header=None)
train1000=pd.read_excel('Proj4Train1000.xlsx',header=None)
test=pd.read_excel('Proj4Test.xlsx',header=None)

x100=train100[[0,1,2,3,4]].values
x1000=train1000[[0,1,2,3,4]].values
y100=train100[[5]].values.reshape(-1)
y1000=train1000[[5]].values.reshape(-1)

X_test=test[[0,1,2,3,4]].values
y_test=test[[5]].values.reshape(-1)

class naivebayes:

    def fit(self, X,Y):
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

    print("naive error:",  accuracy(y_test,predictions))