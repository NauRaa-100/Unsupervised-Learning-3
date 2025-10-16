
# t-SNE Visualization (Unsupervised Learning)

##  Objective
Explore **t-SNE (t-distributed Stochastic Neighbor Embedding)** — a nonlinear dimensionality reduction technique used mainly for **visualizing high-dimensional data** in 2D or 3D.

Unlike PCA, which is **linear** and focuses on variance, t-SNE preserves the **local structure** of the data (neighborhoods and clusters) — making it especially useful for revealing complex patterns or class separations.

---

## Concepts Recap

| Method | Type | Preserves | Typical Use |
|---------|------|------------|--------------|
| **PCA** | Linear | Global Variance | Feature reduction, pre-processing |
| **t-SNE** | Non-linear | Local Relationships | Visualization & clustering insights |

---

##  Key Parameters
| Parameter | Meaning | Notes |
|------------|----------|-------|
| `n_components` | Target dimensions (usually 2 or 3) | For visualization |
| `perplexity` | Roughly the number of neighbors each point considers | Typical range: 5–50 |
| `learning_rate` | Step size during optimization | If too high → scattered, too low → crowded |
| `random_state` | Ensures reproducibility | Always good practice |

---

## Example 1 — MNIST Small (Digits Dataset)

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10,7))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=30, alpha=0.8)
plt.colorbar(scatter, label='Digit Label')
plt.title("t-SNE Visualization of Digits Dataset")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()
````

 **Interpretation:**

* Each color = one digit (0–9).
* Well-separated clusters → good feature representation.
* Overlapping areas → digits that look visually similar.

---

##  Example 2 — Synthetic Clustered Data

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate 3 synthetic clusters
np.random.seed(42)
cluster_1 = np.random.randn(60, 5) + [2, 2, 2, 2, 2]
cluster_2 = np.random.randn(60, 5) + [7, 7, 7, 7, 7]
cluster_3 = np.random.randn(60, 5) + [12, 2, 6, 4, 8]

X = np.vstack([cluster_1, cluster_2, cluster_3])
y = np.array([0]*60 + [1]*60 + [2]*60)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=50, alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.title("t-SNE on Synthetic 3-Cluster Data")
plt.show()
```

 **Observation:**
Distinct, well-separated blobs = good clustering preservation.

---

##  Example 3 — Swiss Roll (Nonlinear Manifold)

```python
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate a 3D nonlinear dataset
X, y = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10,6))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.colorbar(scatter, label='Color = Position along the roll')
plt.title("t-SNE on Swiss Roll Dataset")
plt.show()
```

 **Insight:**
t-SNE successfully **unrolls** the twisted 3D shape into 2D — capturing the structure that PCA might miss.

---

##  Example 4 — Comparing PCA vs t-SNE (Wine Dataset)

```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = load_wine()
X = data.data
y = data.target

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', s=50, alpha=0.8)
plt.title("PCA Projection (2D)")
plt.show()

# t-SNE 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=50, alpha=0.8)
plt.title("t-SNE Projection (2D)")
plt.show()
```

 **Result:**
t-SNE usually reveals clearer **nonlinear separations** than PCA.

---

## Summary

| Aspect  | PCA               | t-SNE                              |
| ------- | ----------------- | ---------------------------------- |
| Type    | Linear            | Non-linear                         |
| Focus   | Global variance   | Local structure                    |
| Output  | Deterministic     | Stochastic (slightly varies)       |
| Purpose | Feature reduction | Visualization                      |
| Speed   | Fast              | Slower (especially for large data) |

---




##  Author Notes

* Always **standardize** data before t-SNE.
* Tune `perplexity` and `learning_rate` carefully.
* Use **PCA first** (to ~50D) if the original data has too many features.

