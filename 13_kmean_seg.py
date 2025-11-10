"""
K-Means Image Segmentation (K=2)
Segments image into two clusters using color/spatial features
"""

# ============================================================================
# CONFIGURATION - All variables at top
# ============================================================================
# Input/Output paths
DEFAULT_IMAGE_PATH = "dataset/high.png"
OUTPUT_BASE_DIR = "figure_out"
OUTPUT_SUBDIR = "kmeans_segmentation"

# Feature extraction parameters
COLORSPACE = "lab"  # 'lab' or 'rgb'
COORDS_WEIGHT = 0.0  # Weight for spatial coordinates (0 disables)

# Clustering parameters
N_CLUSTERS = 2
N_INIT = 10
RANDOM_SEED = 0

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import numpy as np
from PIL import Image

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from KMEAN_ import (
    configure_matplotlib,
    load_image,
    build_features,
    kmeans_segment,
    relabel_by_brightness,
    save_segmentation,
    save_side_by_side,
    save_cluster_panels,
    ensure_output_dir
)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Allow command-line override of defaults
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    colorspace = sys.argv[2] if len(sys.argv) > 2 else COLORSPACE
    coords_weight = float(sys.argv[3]) if len(sys.argv) > 3 else COORDS_WEIGHT
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else RANDOM_SEED

    # Configure matplotlib
    configure_matplotlib()

    # Prepare output directory
    output_dir = ensure_output_dir(OUTPUT_BASE_DIR, OUTPUT_SUBDIR)

    # Load image
    pil_img, arr = load_image(img_path)
    H, W, _ = arr.shape

    # Build features
    X = build_features(arr, colorspace=colorspace, coords_weight=coords_weight)

    # K-means clustering
    labels, centers, inertia = kmeans_segment(X, K=N_CLUSTERS, seed=seed, n_init=N_INIT)

    # Relabel by brightness for consistent visualization
    labels2 = relabel_by_brightness(labels, centers, colorspace=colorspace)

    # Reshape to 2D image space
    mask_2d = labels2.reshape(H, W).astype(np.uint8)

    # Save outputs
    side_by_side_path = os.path.join(output_dir, "result_side_by_side.png")
    mask_path = os.path.join(output_dir, "mask_segmentation.png")
    clusters_path = os.path.join(output_dir, "clusters_separate.png")
    npy_path = os.path.join(output_dir, "mask_segmentation.npy")

    # Save visualizations
    save_side_by_side(side_by_side_path, arr, mask_2d, K=N_CLUSTERS)
    save_segmentation(mask_path, mask_2d, cmap=("gray" if N_CLUSTERS == 2 else "tab10"), K=N_CLUSTERS)
    save_cluster_panels(clusters_path, arr, mask_2d, K=N_CLUSTERS)
    np.save(npy_path, mask_2d)

    # Create colored overlay (normalize mask for overlay)
    mask_normalized = mask_2d.astype(float) / max(1, N_CLUSTERS - 1)
    overlay = (arr * 0.6 + np.stack([mask_normalized] * 3, axis=-1) * 0.4).clip(0, 1)
    overlay_path = os.path.join(output_dir, "overlay.png")
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    # Report results
    print(f"[ok] Inertia: {inertia:.3f}")
    print(f"[ok] Number of clusters: {N_CLUSTERS}")
    print(f"[ok] Saved visualizations:")
    print(f"  {side_by_side_path}")
    print(f"  {mask_path}")
    print(f"  {clusters_path}")
    print(f"  {overlay_path}")
    print(f"  {npy_path}")


if __name__ == "__main__":
    main()
