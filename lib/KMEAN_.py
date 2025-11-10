"""
K-Means clustering utility functions
Shared library for image segmentation and cluster analysis
"""

import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import inspect

# Optional scikit-image for LAB color space
try:
    from skimage.color import rgb2lab
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# ============================================================================
# Matplotlib Configuration
# ============================================================================
def configure_matplotlib():
    """Apply consistent matplotlib style across all scripts"""
    params = {
        'savefig.dpi': 300,
        'figure.dpi': 100,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'serif',
        'font.serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(params)


# ============================================================================
# Image I/O
# ============================================================================
def load_image(path):
    """
    Load an image from disk and convert to normalized array

    Args:
        path: Path to image file

    Returns:
        tuple: (PIL.Image, np.ndarray) - PIL image and normalized [H,W,3] array in [0,1]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return im, arr


# ============================================================================
# Feature Construction
# ============================================================================
def build_features(img_arr, colorspace="lab", coords_weight=0.0):
    """
    Create per-pixel features for clustering

    Args:
        img_arr: Image array [H,W,3] in [0,1]
        colorspace: 'lab' (requires scikit-image) or 'rgb'
        coords_weight: Weight for spatial (x,y) coordinates (0 disables)

    Returns:
        np.ndarray: Feature matrix [H*W, n_features]
    """
    H, W, _ = img_arr.shape

    # Color features
    if colorspace.lower() == "lab":
        if not _HAS_SKIMAGE:
            print("[warn] scikit-image not available; falling back to RGB features.")
            col = img_arr.reshape(-1, 3)
        else:
            lab = rgb2lab(img_arr)  # L in [0,100], a,b roughly [-128,127]
            # Normalize LAB to 0..1 ranges
            L = (lab[..., 0] / 100.0)
            a = (lab[..., 1] + 128.0) / 255.0
            b = (lab[..., 2] + 128.0) / 255.0
            col = np.stack([L, a, b], axis=-1).reshape(-1, 3)
    else:
        col = img_arr.reshape(-1, 3)

    # Optional spatial coordinates
    if coords_weight > 0.0:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs = (xx.astype(np.float32) / max(1, W - 1)) * coords_weight
        ys = (yy.astype(np.float32) / max(1, H - 1)) * coords_weight
        xy = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        X = np.concatenate([col, xy], axis=1)
    else:
        X = col

    return X


# ============================================================================
# K-Means Clustering
# ============================================================================
def kmeans_segment(X, K=2, seed=0, n_init=10):
    """
    Perform K-means clustering

    Args:
        X: Feature matrix [n_samples, n_features]
        K: Number of clusters
        seed: Random seed
        n_init: Number of initializations

    Returns:
        tuple: (labels, centers, inertia)
    """
    km = KMeans(n_clusters=K, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    inertia = km.inertia_
    return labels, centers, inertia


def relabel_by_brightness(labels, centers, colorspace="lab"):
    """
    Reorder labels by brightness (darkest to brightest)

    Args:
        labels: Cluster labels
        centers: Cluster centers
        colorspace: 'lab' or 'rgb'

    Returns:
        np.ndarray: Relabeled cluster assignments (0=darkest, K-1=brightest)
    """
    # Estimate brightness from centroid color channels
    if centers.shape[1] >= 3:
        if colorspace.lower() == "lab":
            brightness = centers[:, 0]  # L channel
        else:  # rgb
            brightness = centers[:, :3].mean(axis=1)
    else:
        brightness = np.arange(centers.shape[0])  # fallback

    # Sort clusters by brightness (ascending)
    order = np.argsort(brightness)

    # Create mapping: old_label -> new_label (0=darkest, ..., K-1=brightest)
    mapping = {old_label: new_label for new_label, old_label in enumerate(order)}

    relabeled = np.vectorize(mapping.get)(labels)
    return relabeled


# ============================================================================
# Sampling
# ============================================================================
def random_sample_idx(n, max_n, seed=0):
    """
    Generate random sample indices

    Args:
        n: Total number of samples
        max_n: Maximum samples to select (None or <=0 returns all)
        seed: Random seed

    Returns:
        np.ndarray: Sorted indices
    """
    if max_n is None or max_n <= 0 or max_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


# ============================================================================
# Cluster Evaluation Metrics
# ============================================================================
def scan_k_metrics(Xs, k_min, k_max, seed=0, n_init=10):
    """
    Scan range of K values and compute clustering metrics

    Args:
        Xs: Feature matrix
        k_min: Minimum K
        k_max: Maximum K
        seed: Random seed
        n_init: Number of K-means initializations

    Returns:
        tuple: (Ks, inertia, silhouette, davies_bouldin, calinski_harabasz)
    """
    Ks, inertia, sil, db, ch = [], [], [], [], []
    for K in range(max(2, k_min), max(2, k_max) + 1):
        km = KMeans(n_clusters=K, n_init=n_init, random_state=seed)
        labels = km.fit_predict(Xs)
        Ks.append(K)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(Xs, labels)))
        db.append(float(davies_bouldin_score(Xs, labels)))
        ch.append(float(calinski_harabasz_score(Xs, labels)))
    return np.array(Ks), np.array(inertia), np.array(sil), np.array(db), np.array(ch)


def elbow_k_by_max_distance(Ks, vals):
    """
    Detect elbow point using maximum distance from line between endpoints

    Args:
        Ks: Array of K values
        vals: Corresponding metric values (e.g., inertia)

    Returns:
        tuple: (best_k, distances) - Recommended K and distance array
    """
    x = Ks.astype(float)
    y = vals.astype(float)
    x = (x - x.min()) / max(1e-12, (x.max() - x.min()))
    y = (y - y.min()) / max(1e-12, (y.max() - y.min()))
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    v = p2 - p1
    vn = v / (np.linalg.norm(v) + 1e-12)

    dists = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        d = np.linalg.norm((p - p1) - np.dot((p - p1), vn) * vn)
        dists.append(d)
    idx = int(np.argmax(dists))
    return int(Ks[idx]), np.array(dists)


def consensus_best_k(Ks, sil, db, ch, elbow_k):
    """
    Determine best K using rank-based consensus across metrics

    Args:
        Ks: Array of K values
        sil: Silhouette scores (higher better)
        db: Davies-Bouldin scores (lower better)
        ch: Calinski-Harabasz scores (higher better)
        elbow_k: Elbow method recommendation

    Returns:
        tuple: (best_k, avg_rank) - Recommended K and average ranks
    """
    # Ranks: lower is better
    r_sil = np.argsort(np.argsort(-sil))  # descending
    r_ch = np.argsort(np.argsort(-ch))    # descending
    r_db = np.argsort(np.argsort(db))     # ascending
    avg_rank = (r_sil + r_ch + r_db) / 3.0

    # Pick minimum average rank; tiebreak with elbow_k, then min K
    best_idx = np.argmin(avg_rank)
    candidate_ks = Ks[avg_rank == avg_rank[best_idx]]
    if elbow_k in candidate_ks:
        best_k = int(elbow_k)
    else:
        best_k = int(np.min(candidate_ks))
    return best_k, avg_rank


# ============================================================================
# t-SNE Embedding
# ============================================================================
def tsne_embed(Xs, seed=0, perplexity=30.0, max_iter=1000):
    """
    Create 2D t-SNE embedding of features

    Args:
        Xs: Feature matrix
        seed: Random seed
        perplexity: t-SNE perplexity parameter
        max_iter: Maximum iterations (replaces deprecated n_iter)

    Returns:
        np.ndarray: 2D embedding [n_samples, 2]
    """
    n = len(Xs)
    if n <= 3:
        raise ValueError("t-SNE needs more than 3 samples.")

    # Clamp perplexity to safe value
    max_perp = max(5.0, min(50.0, (n - 1) / 3.0))
    perp = min(perplexity, max_perp)

    # Build kwargs with only supported parameters
    base_kwargs = dict(
        n_components=2,
        perplexity=perp,
        max_iter=max_iter,  # Use max_iter instead of n_iter
        learning_rate="auto",
        init="pca",
        random_state=seed,
        verbose=0,
        method=("barnes_hut" if n > 1000 else "exact"),
    )

    # Filter to only parameters supported by this sklearn version
    sig = inspect.signature(TSNE.__init__)
    safe_kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

    ts = TSNE(**safe_kwargs)
    return ts.fit_transform(Xs)


# ============================================================================
# Visualization
# ============================================================================
def save_segmentation(figpath_mask, mask_2d, cmap="gray", K=None):
    """
    Save segmentation mask

    Args:
        figpath_mask: Output path
        mask_2d: 2D mask array
        cmap: Colormap
        K: Number of clusters (if None, auto-detect from mask)
    """
    if K is None:
        K = len(np.unique(mask_2d))

    plt.figure(figsize=(6, 6))
    if K == 2:
        plt.imshow(mask_2d, cmap=cmap, vmin=0, vmax=1)
    else:
        plt.imshow(mask_2d, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(figpath_mask, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_side_by_side(figpath, img_arr, mask_2d, K=None):
    """
    Save side-by-side comparison of original image and mask

    Args:
        figpath: Output path
        img_arr: Original image array [H,W,3]
        mask_2d: Segmentation mask [H,W]
        K: Number of clusters (if None, auto-detect from mask)
    """
    if K is None:
        K = len(np.unique(mask_2d))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if K == 2:
        plt.imshow(mask_2d, cmap="gray", vmin=0, vmax=K-1)
    else:
        # Use tab10 for <=10 clusters, viridis for more
        cmap = "tab10" if K <= 10 else "viridis"
        plt.imshow(mask_2d, cmap=cmap, vmin=0, vmax=K-1)
    plt.title(f"K-Means Segmentation (K={K})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_cluster_panels(figpath, img_arr, mask_2d, K=None):
    """
    Save multi-panel visualization showing original image and each cluster separately

    Args:
        figpath: Output path
        img_arr: Original image array [H,W,3]
        mask_2d: Segmentation mask [H,W]
        K: Number of clusters (if None, auto-detect from mask)
    """
    if K is None:
        K = len(np.unique(mask_2d))

    # Determine grid layout
    if K <= 3:
        ncols = K + 1  # Original + clusters in one row
        nrows = 1
    elif K <= 6:
        ncols = 3
        nrows = (K + 2) // 3  # +1 for original, round up
    else:
        ncols = 4
        nrows = (K + 3) // 4  # +1 for original, round up

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    # Show original image
    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis("off")

    # Show each cluster
    for k in range(K):
        cluster_mask = (mask_2d == k).astype(float)
        cluster_img = img_arr * cluster_mask[:, :, np.newaxis]

        axes[k + 1].imshow(cluster_img)
        axes[k + 1].set_title(f"Cluster {k}", fontweight='bold')
        axes[k + 1].axis("off")

    # Hide unused subplots
    for idx in range(K + 1, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_curve(out, Ks, ys, ylabel, highlight_k=None):
    """
    Plot metric vs K curve

    Args:
        out: Output path
        Ks: Array of K values
        ys: Metric values
        ylabel: Y-axis label
        highlight_k: K value to highlight with marker
    """
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(Ks, ys, marker="o")
    if highlight_k is not None and highlight_k in set(Ks):
        i = np.where(Ks == highlight_k)[0][0]
        plt.scatter([Ks[i]], [ys[i]], s=120, edgecolor="k")
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_tsne_scatter(out, Z, labels, title):
    """
    Plot t-SNE scatter with cluster labels

    Args:
        out: Output path
        Z: 2D embedding [n_samples, 2]
        labels: Cluster labels
        title: Plot title
    """
    K = len(np.unique(labels))

    plt.figure(figsize=(8, 6))

    # Choose colormap based on number of clusters
    if K <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.viridis

    # Create scatter plot
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=5, cmap=cmap, alpha=0.6, edgecolors='none')

    # Add colorbar for reference
    cbar = plt.colorbar(scatter, ticks=range(K))
    cbar.set_label('Cluster', rotation=270, labelpad=20)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


# ============================================================================
# Output Management
# ============================================================================
def ensure_output_dir(base_dir, subdir=None):
    """
    Create output directory structure

    Args:
        base_dir: Base output directory
        subdir: Optional subdirectory

    Returns:
        str: Full output path
    """
    if subdir:
        output_path = os.path.join(base_dir, subdir)
    else:
        output_path = base_dir
    os.makedirs(output_path, exist_ok=True)
    return output_path
