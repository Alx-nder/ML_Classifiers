import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.datasets import make_moons

np.random.seed(19680801)

# fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(7,3))


t=np.linspace(0,2*np.pi,300,endpoint=False)
r=np.random.uniform([[2],[4]],[[0],[9]], size=(2,300))
inner=[r[0]*np.cos(t),r[0]*np.sin(t)]
outer=[r[1]*np.cos(t),r[1]*np.sin(t)]

fig1 = plt.figure()
ax1  = fig1.add_subplot()

ax1.scatter(inner[0],inner[1],c="purple",edgecolor='k')
ax1.scatter(outer[0],outer[1],c="yellow",edgecolor='k')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('Scatter plot of random samples')

a=np.array(inner[0]).reshape(-1,1)
b=np.array(inner[1]).reshape(-1,1)
newInner=np.concatenate((a,b,2*(a**2) + 2*(b**2)),axis=1)

c=np.array(outer[0]).reshape(-1,1)
d=np.array(outer[1]).reshape(-1,1)

newOuter=np.concatenate((c,d,2*(c**2) + 2*(d**2)),axis=1)
data=np.concatenate((newInner,newOuter))
y=np.concatenate((np.zeros(len(newInner)),np.ones(len(newOuter))))

samples1=np.concatenate((a,b),axis=1)
samples2=np.concatenate((c,d),axis=1)
samples=np.concatenate((samples1,samples2))

pca = PCA(n_components=1)
n=np.concatenate((a,b),axis=1)
m=np.concatenate((c,d),axis=1)
nm=np.concatenate((n,m))
X_pca = pca.fit_transform(nm)
 
# Plot the results
fig2 = plt.figure()
ax2  = fig2.add_subplot()

ax2.scatter(X_pca[:, 0], [[0.0]*len(n)+[0.05]*len(m)] ,c=y ,alpha=0.5)
ax2.set_ylim(-2,4)
ax2.set_title('regular PCA on orginal samples')
ax2.set_xlabel('Principal Component 1')


fig3d = plt.figure()
ax3d  = fig3d.add_subplot(projection='3d')
ax3d.scatter(inner[0],inner[1],2*(inner[0]**2) + 2*(inner[1]**2),c="purple",edgecolor='k')
ax3d.scatter(outer[0],outer[1],2*(outer[0]**2) + 2*(outer[1]**2),c='yellow',edgecolor='k')
ax3d.set_title('samples embedded in a higher dimension')


# Apply PCA with two components (for 2D visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)
 
# Plot the results
fig4 = plt.figure()
ax4  = fig4.add_subplot()

ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
ax4.set_title('two PCA of embedded samples')
ax4.set_xlabel('Principal Component 1')
ax4.set_ylabel('Principal Component 2')

# Apply PCA with one components (visualization)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(data)
 
# Plot the results
fig5 = plt.figure()
ax5  = fig5.add_subplot()

ax5.scatter(X_pca[:, 0], np.zeros(len(X_pca[:, 0])) ,c=y )
ax5.set_title('one PCA of embedded samples')
ax5.set_xlabel('Principal Component 1')


def gaus_kpca(X, gamma, n_components):

    # Computes the squared Euclidean distance between the points in the matrix.
    # Converting the pairwise distances into a symmetric matrix.
    mat_sq_dists = squareform(pdist(X, 'sqeuclidean'))

    # Computing the MxM kernel matrix.
    K = np.exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
   
    eigvals, eigvecs = eigh(K)

    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]    

    # Collect the top k eigenvectors (projected examples)

    X_pc = np.column_stack([eigvecs[:, i]for i in range(n_components)]) 
    return X_pc

X, y = make_moons(100, random_state=123)

fig6 = plt.figure()
ax6  = fig6.add_subplot()

ax6.scatter(X[y==0, 0], X[y==0, 1],color='yellow', marker='^')
ax6.scatter(X[y==1, 0], X[y==1, 1],color='purple', marker='o')
ax6.set_title('New non separable data points')

X_pc = gaus_kpca(samples, gamma=14, n_components=1)

fig7 = plt.figure()
ax7  = fig7.add_subplot()

X_kpca = gaus_kpca(X, gamma=15, n_components=2)
ax7.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='yellow', marker='^')
ax7.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='purple', marker='o')
ax7.set_xlabel('PC1')
ax7.set_ylabel('PC2')
ax7.set_title('Gaussian 2 PCA on samples')

fig8 = plt.figure()
ax8  = fig8.add_subplot()

ax8.scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,color='yellow', marker='^')
ax8.scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,color='purple', marker='o')
ax8.set_ylim(-2,4)
ax8.set_xlabel('PC1')
ax8.set_title('Gaussian 1 PCA on samples')

plt.show()