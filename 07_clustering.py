import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from lib.util import high_contrast_colors, compute_kmeans, build_code_to_name, map_clusters_to_facies, scatter_plot

# -----------------------
# Config & data loading
# -----------------------
csv_path = 'dataset/well_log.csv'   # change if needed
data = pd.read_csv(csv_path)
data = data.iloc[::10].copy()       # take every 10th point

# Features
x = data['GR'].to_numpy()
y = data['ILD_log10'].to_numpy()

# Facies names 
lithofacies = ['SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS']

# Optional custom colors aligned to lithofacies (same order)
lithocolors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72',
               '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Number of clusters guided by unique facies in the CSV
number_of_clusters = int(pd.Series(data['Facies']).nunique())

labels, centers = compute_kmeans(x, y, number_of_clusters, random_state=42, reorder_by='GR')

# Optional clustering quality
X = np.column_stack([x, y])
sil = silhouette_score(StandardScaler().fit_transform(X), labels)
print(f"Silhouette score (z-scored features): {sil:.3f}")

cluster_to_name, name_to_color = map_clusters_to_facies(
    labels, data['Facies'], lithofacies, lithocolors
)

scatter_plot(
    x, y, labels, centers,
    cluster_to_name=cluster_to_name,
    name_to_color=name_to_color,
    title='Well Log Clusters (Facies)'
)

# Inspect outputs
centroids_df = pd.DataFrame(centers, columns=['GR_center', 'ILD_log10_center'])
centroids_df.index.name = 'ClusterID (0-based after optional reordering)'
print("\nCentroids (original units):\n", centroids_df)

print("\nCluster → Facies mapping:")
for cid in sorted(cluster_to_name):
    print(f"  Cluster {cid} -> {cluster_to_name[cid]}")
