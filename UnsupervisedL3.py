"""
Unsupervised Learning ..
-- t-SNE visualization (MNIST small).
-- Clustering evaluation (Elbow method, Silhouette score).

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data   
y = digits.target 
print("Original shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

print("t-SNE shape:", X_tsne.shape)

plt.figure(figsize=(10,7))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=30, alpha=0.8)
plt.colorbar(scatter, label='Digit Label')
plt.title("t-SNE Visualization of Digits Dataset")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

print('-------------Seperate-----------') 

# Create synthetic 3-cluster data
np.random.seed(42)
cluster_1 = np.random.randn(60, 5) + [2, 2, 2, 2, 2]
cluster_2 = np.random.randn(60, 5) + [7, 7, 7, 7, 7]
cluster_3 = np.random.randn(60, 5) + [12, 2, 6, 4, 8]

X = np.vstack([cluster_1, cluster_2, cluster_3])
y = np.array([0]*60 + [1]*60 + [2]*60)

# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Apply t-SNE (complete yourself)
# visualize 2D points (use color = y)

tsne=TSNE(n_components=2,perplexity=30,learning_rate=200,random_state=42)
xtsn=tsne.fit_transform(x_scaled)

scatter=plt.scatter(xtsn[:,0],xtsn[:,1],c=y,cmap='tab10',alpha=0.7,s=50)
plt.colorbar(scatter,label='Target')
plt.show()

print('-------------Seperate-----------') 


from sklearn.datasets import make_swiss_roll

# Generate a non-linear 3D dataset
x, y = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
x = pd.DataFrame(x, columns=['x1', 'x2', 'x3'])

# Compare the 2D projection with color-coded "y"

scale=StandardScaler()
x_scaled=scale.fit_transform(x)

tsne=TSNE(n_components=2,perplexity=50,learning_rate=200,random_state=42)
xtsn=tsne.fit_transform(x_scaled)
plt.figure(figsize=(10,6))
sc=plt.scatter(xtsn[:,0],xtsn[:,1],c=y,cmap='tab10',s=50,alpha=0.8)
plt.colorbar(sc,label='Target')
plt.show()
print('-------------Seperate-----------') 

from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

data = load_wine()
x = data.data
y = data.target

scale=StandardScaler()
x_scaled=scale.fit_transform(x)

pca=PCA(n_components=None,random_state=42)
pca.fit(x_scaled)

var_ratio=pca.explained_variance_ratio_
cummulative=np.cumsum(var_ratio)

plt.plot(range(1,len(cummulative)+1),cummulative)
plt.axhline(y=0.9,linestyle='--',c='r')
plt.show()

n=np.argmax(cummulative >= 0.9) +1
pca=PCA(n_components=n,random_state=42)
xpc=pca.fit_transform(x_scaled)

sc=plt.scatter(xpc[:,0],xpc[:,1],c=y,cmap='tab10',alpha=0.8,s=50)
plt.colorbar(sc,label='Target')
plt.show()


tsne=TSNE(n_components=2,random_state=42)
xts=tsne.fit_transform(x_scaled)

sct=plt.scatter(xts[:,0],xts[:,1],c=y,cmap='tab10',alpha=0.8,s=50)
plt.colorbar(sct,label='target')
plt.show()

print('-------------Seperate-----------') 

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=4, n_features=6, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

perplexities = [5, 20, 50, 100]

plt.figure(figsize=(12,10))

for i, p in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=p, learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.subplot(2,2,i+1)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=40, alpha=0.8)
    plt.title(f"t-SNE with perplexity={p}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

plt.tight_layout()
plt.show()
