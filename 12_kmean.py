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

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

from lib.KMEAN_ import configure_matplotlib

# Output folder (one folder per script, named after the script)
OUTDIR = Path("12_kmean")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Configure plotting style (shared global plot style)
    configure_matplotlib()

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

        plt.figure()  # global figure.figsize default
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
        plt.tight_layout()
        output_path = OUTDIR / f"kmeans_basic_K{N_CLUSTERS}.png"
        plt.savefig(output_path, format="png", bbox_inches="tight")
        print(f"[ok] Saved plot to: {output_path}")

        plt.show()
        plt.close()

    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
