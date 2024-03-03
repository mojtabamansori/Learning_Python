import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
from scipy.stats import multivariate_normal

forest_fires_path = 'forest_fires.csv'
forest_fires = pd.read_csv(forest_fires_path)
forest_fires['target'] = np.where(forest_fires['Classes'] == 'fire', 1, 0)
selected_features = forest_fires[['Temperature', 'RH', 'Ws', 'Rain', 'Region']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features)

# Hierarchical clustering
linkage_matrix = linkage(scaled_features, method='ward')

# Determine the optimal number of clusters
k = 2  # Set the desired number of clusters
clusters = fcluster(linkage_matrix, k, criterion='maxclust')
forest_fires['cluster'] = clusters

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Dendrogram
dendrogram(linkage_matrix, truncate_mode='lastp', p=15, orientation='top', show_leaf_counts=True, ax=ax1)
ax1.set_title('Hierarchical Clustering Dendrogram')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')

# Cluster plot
sns.scatterplot(x='Temperature', y='RH', hue='cluster', data=forest_fires, palette='Set1', ax=ax2)
ax2.set_title('Hierarchical Clustering of Forest Fires')

# Plot the cluster centers on the cluster plot with Density Contours
num_clusters = max(clusters)
for i in range(1, num_clusters + 1):
    cluster_data = forest_fires[forest_fires['cluster'] == i][['Temperature', 'RH']]

    # Create a 2D grid for the contour plot
    x, y = np.meshgrid(np.linspace(cluster_data['Temperature'].min(), cluster_data['Temperature'].max(), 100),
                       np.linspace(cluster_data['RH'].min(), cluster_data['RH'].max(), 100))

    mvn = multivariate_normal(mean=cluster_data.mean().values, cov=np.cov(cluster_data, rowvar=False))
    z = mvn.pdf(np.vstack([x.ravel(), y.ravel()]).T)
    z = z.reshape(x.shape)
    ax2.contourf(x, y, z, cmap='Reds', alpha=0.3)

    # Plot Cluster Centers
    ax2.scatter(cluster_data.mean().values[0], cluster_data.mean().values[1], c='red', marker='D', s=100,
                label=f'Cluster {i} Center')

ax2.set_title('Hierarchical Clustering of Forest Fires with Centers and Density Contours')

# True labels
sns.scatterplot(x='Temperature', y='RH', hue='target', data=forest_fires, palette='viridis', ax=ax3)
ax3.set_title('True Labels')

# Plot True Label Centers and Density Contours
for i in range(num_clusters):
    cluster_data = forest_fires[forest_fires['target'] == i][['Temperature', 'RH']]

    # Create a 2D grid for the contour plot
    x, y = np.meshgrid(np.linspace(cluster_data['Temperature'].min(), cluster_data['Temperature'].max(), 100),
                       np.linspace(cluster_data['RH'].min(), cluster_data['RH'].max(), 100))

    mvn = multivariate_normal(mean=cluster_data.mean().values, cov=np.cov(cluster_data, rowvar=False))
    z = mvn.pdf(np.vstack([x.ravel(), y.ravel()]).T)
    z = z.reshape(x.shape)
    ax3.contourf(x, y, z, cmap='Blues', alpha=0.3)

    # Plot True Label Centers
    ax3.scatter(cluster_data.mean().values[0], cluster_data.mean().values[1], c='blue', marker='D', s=100,
                label=f'True Label {i} Center')

plt.show()
