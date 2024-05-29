import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def multi_least_squares(feature):
    names={'0':'setosa','1':'versicolor','2':'virginica'}
    df=pd.read_excel("Pattern_Recognition\\proj1\\Proj1DataSet.xlsx")

    counts=df.species.value_counts().values
    t1=[[1,0,0]]*counts[0]
    t2=[[0,1,0]]*counts[1]
    t3=[[0,0,1]]*counts[2]
    T=t1+t2+t3

    scaler=StandardScaler()


    # add offset and drop species to get feature variables fv or X
    df.drop(columns='species',inplace=True)

    # scale data
    df_scaled=scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)

    df=df.assign(offset=[1]*df.shape[0])

    feature=[f-1 for f in feature]
    feature.append(-1)
    needed_features = [df.columns[x] for x in feature ]

    fv=df[needed_features].values.tolist()

    #Xt: transpose matrix X
    Xt=np.array(fv).transpose()

    #XtX: matrix multiplication
    XtX=np.matmul(Xt,fv)

    #XtX_inv: inverse of XtX
    XtX_inv=np.linalg.inv(XtX)

    #Xfin (XtX)**-1 multiplied by Xt
    Xfin= np.matmul(XtX_inv,Xt)

    W=np.matmul(Xfin,T)

    decisions=W.transpose()

    # test
    misclassified=0
    for i in range(len(fv)):
        # position of decision function that holds the max d(x) result
        pos=0
        init_dmax=decisions[pos].dot(fv[i])
        for trial in range(len(feature)):
            if decisions[trial].dot(fv[i])>=init_dmax:
                init_dmax=decisions[trial].dot(fv[i])
                pos=trial
    
        # if position does not point to 1
        if T[i][pos]!=max(T[i]):
            misclassified+=1
            print(pos)
    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))

        # 2d
        ax=plt.axes()
        # function
        d1=decisions[0]
        d2=decisions[1]
        d3=decisions[2]
        l1=np.array(d1)-np.array(d2)
        l2=np.array(d1)-np.array(d3)
        l3=np.array(d2)-np.array(d3)
        x=np.linspace(-3, 9, 1000)

        plt.plot(x,-1*(l1[0]*x )/l1[1], label='d1-d2')
        plt.plot(x,-1*(l2[0]*x )/l2[1], label='d1-d3')
        plt.plot(x,-1*(l3[0]*x )/l3[1], label='d2-d3')

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
        plt.ylim(-2,2)
        plt.xlim(-2,3)
        plt.legend(loc="lower right")

        plt.show()

    return W,misclassified

print(multi_least_squares([3,4]))