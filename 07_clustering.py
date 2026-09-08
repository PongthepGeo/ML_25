import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from lib.control_plot import PLOT_PARAMS  # shared global plot style
from lib.util import high_contrast_colors, compute_kmeans, build_code_to_name, map_clusters_to_facies, scatter_plot

matplotlib.rcParams.update(PLOT_PARAMS)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("07_clustering")
OUTDIR.mkdir(parents=True, exist_ok=True)

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
    OUTDIR=OUTDIR,
    title='Well Log Clusters (Facies)'
)

# Inspect outputs
centroids_df = pd.DataFrame(centers, columns=['GR_center', 'ILD_log10_center'])
centroids_df.index.name = 'ClusterID (0-based after optional reordering)'
print("\nCentroids (original units):\n", centroids_df)

centroids_path = OUTDIR / 'centroids.csv'
centroids_df.to_csv(centroids_path)
print('Saved table:', centroids_path)

print("\nCluster -> Facies mapping:")
for cid in sorted(cluster_to_name):
    print(f"  Cluster {cid} -> {cluster_to_name[cid]}")
