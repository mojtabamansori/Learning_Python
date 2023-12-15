import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import multivariate_normal

forest_fires_path = 'forest_fires.csv'
forest_fires = pd.read_csv(forest_fires_path)
forest_fires['target'] = np.where(forest_fires['Classes'] == 'fire', 1, 0)
print(forest_fires.columns)
selected_features = forest_fires[['Temperature', 'RH', 'Ws', 'Rain', 'Region']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features)

num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
forest_fires['cluster'] = kmeans.fit_predict(scaled_features)

# Inverse transform cluster centers to the original scale
cluster_centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot for K-Means Clustering
sns.scatterplot(x='Temperature', y='RH', hue='cluster', data=forest_fires, palette='Set1', ax=ax1)
ax1.scatter(cluster_centers_original_scale[:, 0], cluster_centers_original_scale[:, 1], c='red', marker='X', s=100,
            label='Cluster Centers')

for i in range(num_clusters):
    cluster_data = forest_fires[forest_fires['cluster'] == i][['Temperature', 'RH']]

    # Create a 2D grid for the contour plot
    x, y = np.meshgrid(np.linspace(cluster_data['Temperature'].min(), cluster_data['Temperature'].max(), 100),
                       np.linspace(cluster_data['RH'].min(), cluster_data['RH'].max(), 100))

    mvn = multivariate_normal(mean=cluster_data.mean().values, cov=np.cov(cluster_data, rowvar=False))
    z = mvn.pdf(np.vstack([x.ravel(), y.ravel()]).T)
    z = z.reshape(x.shape)
    ax1.contourf(x, y, z, cmap='Reds', alpha=0.3)

ax1.set_title('K-Means Clustering of Forest Fires with Centers and Density Contours')

# Plot for True Labels with Density Contours
sns.scatterplot(x='Temperature', y='RH', hue='target', data=forest_fires, palette='viridis', ax=ax2)
ax2.set_title('True Labels')

# Plot True Label Centers and Density Contours
for i in range(num_clusters):
    cluster_data = forest_fires[forest_fires['target'] == i][['Temperature', 'RH']]

    # Create a 2D grid for the contour plot
    x, y = np.meshgrid(np.linspace(cluster_data['Temperature'].min(), cluster_data['Temperature'].max(), 100),
                       np.linspace(cluster_data['RH'].min(), cluster_data['RH'].max(), 100))

    mvn = multivariate_normal(mean=cluster_data.mean().values, cov=np.cov(cluster_data, rowvar=False))
    z = mvn.pdf(np.vstack([x.ravel(), y.ravel()]).T)
    z = z.reshape(x.shape)
    ax2.contourf(x, y, z, cmap='Blues', alpha=0.3)

    # Plot True Label Centers
    ax2.scatter(cluster_data.mean().values[0], cluster_data.mean().values[1], c='blue', marker='D', s=100,
                label=f'True Label {i} Center')

plt.show()
