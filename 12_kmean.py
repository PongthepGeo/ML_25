"""
Basic K-Means Clustering Example
Simple demonstration with toy dataset
"""

# ============================================================================
# CONFIGURATION - All variables at top
# ============================================================================
# Dataset configuration
TOY_DATA = [[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [4, 6]]

# Clustering parameters
N_CLUSTERS = 2
RANDOM_SEED = 0

# Output configuration
OUTPUT_BASE_DIR = "figure_out"
OUTPUT_SUBDIR = "kmeans_basic"

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from KMEAN_ import configure_matplotlib, ensure_output_dir

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Configure plotting style
    configure_matplotlib()

    # Prepare output directory
    output_dir = ensure_output_dir(OUTPUT_BASE_DIR, OUTPUT_SUBDIR)

    # Convert toy dataset to numpy array
    X = np.array(TOY_DATA)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(X)

    # Display results
    print("Cluster centers:\n", kmeans.cluster_centers_)
    print("Labels:", labels)
    print("Inertia (sum of sq. dists):", kmeans.inertia_)

    # Visualization
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Use colormap for multiple clusters
        if N_CLUSTERS <= 10:
            colors = cm.tab10(labels)
        else:
            colors = cm.viridis(labels / N_CLUSTERS)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors='k', linewidths=0.5)
        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker="*",
            s=500,
            c='red',
            edgecolors="k",
            linewidths=2,
            label='Centroids'
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"K-Means Clustering (K={N_CLUSTERS})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save figure
        output_path = os.path.join(output_dir, f"kmeans_basic_K{N_CLUSTERS}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[ok] Saved plot to: {output_path}")

        plt.show()
        plt.close()

    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
