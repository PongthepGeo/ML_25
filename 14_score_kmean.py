"""
K-Means Cluster Selection & t-SNE Visualization
Scans K values, computes metrics, and creates t-SNE embeddings
"""

# ============================================================================
# CONFIGURATION - All variables at top
# ============================================================================
# Input paths
DEFAULT_IMAGE_PATH = "dataset/high.png"

# Feature extraction parameters
COLORSPACE = "lab"  # 'lab' or 'rgb'
COORDS_WEIGHT = 0.0  # Weight for spatial coordinates (0 disables)

# Clustering parameters
K_MIN = 2
K_MAX = 10
N_INIT = 10
RANDOM_SEED = 0

# Sampling parameters
MAX_SAMPLE_SIZE = 10000  # Max pixels for metrics & t-SNE

# t-SNE parameters
TSNE_PERPLEXITY = 30.0
TSNE_MAX_ITER = 1000

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from lib.KMEAN_ import (
    configure_matplotlib,
    load_image,
    build_features,
    random_sample_idx,
    scan_k_metrics,
    elbow_k_by_max_distance,
    consensus_best_k,
    tsne_embed,
    plot_metric_curve,
    plot_tsne_scatter,
)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("14_score_kmean")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Allow command-line override of defaults
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    colorspace = sys.argv[2] if len(sys.argv) > 2 else COLORSPACE
    coords_weight = float(sys.argv[3]) if len(sys.argv) > 3 else COORDS_WEIGHT
    k_min = int(sys.argv[4]) if len(sys.argv) > 4 else K_MIN
    k_max = int(sys.argv[5]) if len(sys.argv) > 5 else K_MAX
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else RANDOM_SEED

    # Configure matplotlib (shared global plot style)
    configure_matplotlib()

    # Load image and build features
    _, arr = load_image(img_path)
    X = build_features(arr, colorspace=colorspace, coords_weight=coords_weight)

    # Subsample for metrics & t-SNE (silhouette is O(n^2))
    idx = random_sample_idx(len(X), MAX_SAMPLE_SIZE, seed=seed)
    Xs = X[idx]

    # Scan K and compute metrics
    Ks, inertia, sil, db, ch = scan_k_metrics(Xs, k_min, k_max, seed=seed, n_init=N_INIT)
    elbow_k, elbow_d = elbow_k_by_max_distance(Ks, inertia)
    best_k, avg_rank = consensus_best_k(Ks, sil, db, ch, elbow_k)

    # t-SNE embedding with best K
    km_best = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=seed).fit(Xs)
    Z = tsne_embed(Xs, seed=seed, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_MAX_ITER)

    # Save metrics CSV
    metrics_path = OUTDIR / "k_metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("K,inertia,silhouette,davies_bouldin,calinski_harabasz,avg_rank,elbow_distance\n")
        for i, K in enumerate(Ks):
            f.write(f"{K},{inertia[i]},{sil[i]},{db[i]},{ch[i]},{avg_rank[i]},{elbow_d[i]}\n")

    # Save metric plots
    plot_metric_curve(
        OUTDIR / "elbow_inertia.png",
        Ks, inertia,
        "Inertia (lower better)",
        highlight_k=elbow_k
    )
    plot_metric_curve(
        OUTDIR / "silhouette.png",
        Ks, sil,
        "Silhouette (higher better)",
        highlight_k=best_k
    )
    plot_metric_curve(
        OUTDIR / "davies_bouldin.png",
        Ks, db,
        "Davies-Bouldin (lower better)",
        highlight_k=best_k
    )
    plot_metric_curve(
        OUTDIR / "calinski_harabasz.png",
        Ks, ch,
        "Calinski-Harabasz (higher better)",
        highlight_k=best_k
    )

    # Save t-SNE scatter
    tsne_path = OUTDIR / f"tsne_scatter_K{best_k}.png"
    plot_tsne_scatter(tsne_path, Z, km_best.labels_, f"t-SNE (sample={len(Xs)}) - K={best_k}")

    # Save summary JSON and text
    summary = {
        "best_k": int(best_k),
        "elbow_k": int(elbow_k),
        "k_range": [int(Ks.min()), int(Ks.max())],
        "sample_size": int(len(Xs)),
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_max_iter": TSNE_MAX_ITER,
        "seed": seed,
        "colorspace": colorspace,
        "coords_weight": coords_weight,
        "img": img_path,
    }
    with open(OUTDIR / "best_k.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUTDIR / "best_k.txt", "w") as f:
        f.write(str(best_k) + "\n")

    # Report results
    print("[ok] Saved outputs to:", OUTDIR)
    print("[ok] elbow_k =", elbow_k, "| best_k (consensus) =", best_k)
    print("[ok] t-SNE scatter:", tsne_path)
    print("[ok] metrics CSV:", metrics_path)


if __name__ == "__main__":
    main()
