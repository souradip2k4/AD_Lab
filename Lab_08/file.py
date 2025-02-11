import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = 'Mall_Customers.csv'
data = pd.read_csv(file_path)

features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering with 3 clusters (you can change this value)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)

# Get cluster labels and centroids
data['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the clusters with optimized visualization
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']
for i in range(3):
  cluster_data = scaled_features[data['Cluster'] == i]
  plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], marker=markers[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')

# Plot centroids with enhanced visibility
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=250, label='Centroids', edgecolors='white')

plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('K-Means Clustering of Customers')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Analyze each cluster
for i in range(3):
  cluster = data[data['Cluster'] == i]
  print(f"Cluster {i} Analysis:")
  print(f"Average Annual Income: {cluster['Annual Income (k$)'].mean():.2f}")
  print(f"Average Spending Score: {cluster['Spending Score (1-100)'].mean():.2f}")
  print("-" * 30)