import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import multivariate_normal

forest_fires_path = 'forest_fires.csv'
forest_fires = pd.read_csv(forest_fires_path)
forest_fires['target'] = np.where(forest_fires['Classes'] == 'fire', 1, 0)
selected_features = forest_fires[['Temperature', 'RH', 'Ws', 'Rain', 'Region']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features)

best_eps = None
best_min_samples = None
best_num_clusters = 0

for eps in np.arange(0.1, 1.0, 0.1):
    for min_samples in range(2, 10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        forest_fires['cluster'] = dbscan.fit_predict(scaled_features)

        num_clusters = len(np.unique(forest_fires['cluster'])) - 1

        if num_clusters == 2 and num_clusters > best_num_clusters:
            best_num_clusters = num_clusters
            best_eps = eps
            best_min_samples = min_samples

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
forest_fires['cluster'] = dbscan.fit_predict(scaled_features)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x='Temperature', y='RH', hue='cluster', data=forest_fires, palette='Set1', ax=ax1)
ax1.set_title('DBSCAN Clustering of Forest Fires')

sns.scatterplot(x='Temperature', y='RH', hue='target', data=forest_fires, palette='viridis', ax=ax2)
ax2.set_title('True Labels')

for i in forest_fires['cluster'].unique():
    if i != -1:
        cluster_data = forest_fires[forest_fires['cluster'] == i][['Temperature', 'RH']]
        x, y = np.meshgrid(np.linspace(cluster_data['Temperature'].min(), cluster_data['Temperature'].max(), 100),
                           np.linspace(cluster_data['RH'].min(), cluster_data['RH'].max(), 100))
        mvn = multivariate_normal(mean=cluster_data.mean().values, cov=np.cov(cluster_data, rowvar=False))
        z = mvn.pdf(np.vstack([x.ravel(), y.ravel()]).T)
        z = z.reshape(x.shape)
        ax1.contourf(x, y, z, cmap='Reds', alpha=0.3)

for i in range(2):
    cluster_data = forest_fires[forest_fires['target'] == i][['Temperature', 'RH']]
    x, y = np.meshgrid(np.linspace(cluster_data['Temperature'].min(), cluster_data['Temperature'].max(), 100),
                       np.linspace(cluster_data['RH'].min(), cluster_data['RH'].max(), 100))
    mvn = multivariate_normal(mean=cluster_data.mean().values, cov=np.cov(cluster_data, rowvar=False))
    z = mvn.pdf(np.vstack([x.ravel(), y.ravel()]).T)
    z = z.reshape(x.shape)
    ax2.contourf(x, y, z, cmap='Blues', alpha=0.3)
    ax2.scatter(cluster_data.mean().values[0], cluster_data.mean().values[1], c='blue', marker='D', s=100,
                label=f'True Label {i} Center')

plt.show()
