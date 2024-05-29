import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def leastSquares(target,feature):
    names={'0':'setosa','1':'versicolor','2':'virginica'}

    feature=[f-1 for f in feature]
    feature.append(-1)

    df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")
    
    # add offset and drop species to get feature variables fv or X
    df=df.assign(offset=[1]*df.shape[0])

    needed_features = [df.columns[x] for x in feature ]

    class_size=df.loc[df.species==df.species.unique()[0]].shape[0]

    # target vector ... order matters
    if target[0]==1:
        t=[0]*class_size + [1]*(df.shape[0]-class_size)
    elif target[0]==2:
        t=[1]*class_size + [0]*class_size +[1]*class_size
    elif target[0]==3:
        t= [1]*(df.shape[0]-class_size)+ [0]*class_size

    # feature vectors fv or X extracted from dataframe
    fv=df[needed_features].values.tolist()

    #Xt: transpose matrix X
    Xt=np.array(fv).transpose()

    #XtX: matrix multiplication
    XtX=np.matmul(Xt,fv)

    #XtX_inv: inverse of XtX
    XtX_inv=np.linalg.inv(XtX)

    #X final (XtX)**-1 multiplied by Xt
    Xfin= np.matmul(XtX_inv,Xt)

    w=np.matmul(Xfin,t)

    # test
    misclassed=0
    for x in range(len(fv)):
        # first class <= 0.5, second class >= 0.5
        if (w.dot(fv[x])<0.5 and (t[x]==1)) or (w.dot(fv[x])>=0.5 and (t[x]==0)) or ():
            misclassed+=1

    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))
        ax=plt.axes()

        w=w.tolist()
        # 2d
        
        x = np.array([0,1, 2, 3,4,5,6,7,8,9 ])  # X-axis points
        y = -(x * w[0] +w[2])/x*w[1]  # Y-axis points
    
        plt.plot(x, y)  # Plot the chart

        fv=np.array(fv)
        fv=fv.T

        c1=list(zip(fv[0][0:50],fv[1][0:50]))
        x,y=zip(*c1)
        ax.scatter(x,y,color='green',label=names['0'])
        
        c2=list(zip(fv[0][50:100],fv[1][50:100]))
        x,y=zip(*c2)
        ax.scatter(x,y,color='red',label=names['1'])
        
        c3=list(zip(fv[0][100:150],fv[1][100:150]))
        x,y=zip(*c3)
        ax.scatter(x,y,color='orange',label=names['2'])

    # function to show the plot
        plt.ylim(0,3)
        plt.xlim(0,9)
        plt.legend(loc="lower right")

        plt.show()
    
    return w,misclassed

print(leastSquares([1,2,3],[3,4]))