
w=[1.0000000e+00, 8.8817842e-16 ,1.00000000e+00]
classes = [[[0,0],[0,1]], [[1,0],[1,1]]]
ro=1



def dotProduct(wv,fv,c):
    if c==2:
        fv=[-v for v in fv]
    res=0
    for i,j in zip(wv, fv):
        res+=i*j
    return res

def newWeight(wv,fv,c,ro):
    if c==2:
        fv=[-v for v in fv]
    newWV=[w+(ro*v) for w,v in zip(wv,fv)]
    return newWV

wv=[]
counter =0
wv.append(w)
while counter < len(classes[0])+len(classes[1]) +1:
    print("streak",counter)
    for c in range(len(classes)):
        for fv in classes[c]:
            if len(fv)<len(w):
                fv.append(1)

            if dotProduct(w,fv,c+1)<=0:
                w=newWeight(w,fv,c+1,ro)
                wv.append(w)
                counter=0 
    counter+=1
print(wv[-1])
w=wv[-1]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig=plt.figure(figsize=(5,5))
# ax=plt.axes(projection='3d')
ax=plt.axes()

# ax=fig.add_subplot(111,projection='3d')

# xx,yy=np.meshgrid(range(-2,3),range(-2,3))
# z=(w[0]*xx + w[1]*yy)/w[2]

# ax.plot_surface(xx, yy, z, alpha=0.5)
x=np.linspace(-1, 2, 1000)
    # function
plt.plot(x,-1*(w[0]*x + w[2])/w[1])

ax.scatter([0,0],[0,1],color='green')
ax.scatter([1,1],[0,1], color='red')

# function to show the plot
plt.show()