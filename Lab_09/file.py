import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratios: {explained_variance}")
print(f"Total Variance Captured: {sum(explained_variance):.2f}")

plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
  plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Wine Dataset")
plt.legend()
plt.savefig("2d-projection.png")
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for i, target_name in enumerate(target_names):
  ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=target_name)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("3D PCA Projection of Wine Dataset")
ax.legend()
plt.savefig("3d-projection.png")
plt.show()

print("Souradip Saha")
print("220529392")