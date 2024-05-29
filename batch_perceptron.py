import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def batch_perceptron(target,feature, ro):
    names={'0':'setosa','1':'versicolor','2':'virginica'}

    target=[t-1 for t in target]
    feature=[f-1 for f in feature] 
    feature.append(-1)
    
    df=pd.read_excel("Pattern_Recognition\\proj1\\Proj1DataSet.xlsx")
    # add offset column 
    df=df.assign(offset=[1]*df.shape[0])

    # separate features we need
    needed_features = [df.columns[x] for x in feature ]

    classes=[(df.loc[df['species'] ==fv])[needed_features].values.tolist() for fv in df.species.unique()]
    # create class 0, class 1
    classes=[classes[target[0]],classes[target[1]]+classes[target[2]]]

    # class 2 negative
    classes[1]=-1*np.array(classes[1])

    # initial guess for weight vector: all zeros
    w=np.array([0]*len(feature))

    count=0
    while count!=-1:
        # epoch
        # reset misclassified features
        misclas_fv=[0]*len(w)
        
        for c in range(len(classes)):
            for fv in classes[c]:
                
                if w.dot(fv)<=0:
                    misclas_fv=np.add(fv,misclas_fv)
        # if no misclassification occurs stop running
        if list(misclas_fv) == ([0]*len(list(w))):
            break
        # if 50 epoch passes, stop running
        elif count==50:
            break        
        else:
            # otherwise update weight vector
            w=np.add(w, (ro*np.array(misclas_fv)))
            count+=1

    # test
    misclassed=0
    for c in range(len(classes)):
        for fv in classes[c]:    
            if w.dot(fv)<=0:
                misclassed+=1

    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))
        ax=plt.axes()
        
        x=np.linspace(0, 9, 1000)
        plt.plot(x,-(w[0]*x+w[2])/w[1])

        c1=np.array(classes[0]).T
        c1=list(zip(c1[0],c1[1]))
        x,y=zip(*c1)
        ax.scatter(x,y,color='green',label=names[f'{target[0]}'])
        
        c2=-1*np.array(classes[1][:50]).T
        c2=list(zip(c2[0],c2[1]))
        x,y=zip(*c2)
        ax.scatter(x,y,color='red',label=names[f'{target[1]}'])

        c3=-1*np.array(classes[1][50:]).T
        c3=list(zip(c3[0],c3[1]))
        x,y=zip(*c3)
        ax.scatter(x,y,color='orange',label=names[f'{target[2]}'])

    # function to show the plot
        plt.ylim(0,3)
        plt.xlim(0,9)
        plt.legend(loc="upper left")
        plt.show()
    
    return(count,w,misclassed)

print(batch_perceptron([3,2,1],[3,4],.5))
