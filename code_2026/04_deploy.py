import argparse
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from matplotlib.patches import Patch

# ============================================================================
# Shared code in main_code/lib: plot style and segmentation colors
# (control_plot.py), model loading + metrics (mlp_util.py, also used by 03_mlp.py)
# ============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))

from control_plot import (
    PLOT_PARAMS, BACKGROUND_VALUE, SOIL_VALUE, COLOR_BACKGROUND, COLOR_SOIL,
    COLOR_FALSE_POSITIVE, COLOR_FALSE_NEGATIVE,
)
from mlp_util import load_model, summarize

plt.rcParams.update(PLOT_PARAMS)

# ============================================================================
# WHOLE-IMAGE DEPLOYMENT
# Load the model trained by 03_mlp.py and predict EVERY pixel of the test image
# (the table made from image 2 by 01_image_to_table.py). No training happens here.
# ============================================================================
# Paths are relative to this script, so it runs from any folder.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / '03_mlp' / 'mlp_model.pth'                   # made by 03_mlp.py
DEFAULT_TABLE = SCRIPT_DIR / '01_image_to_table' / 'test.csv'             # made by 01_image_to_table.py
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / '04_deploy'

PRED_MASK_FILE = 'predicted_mask.png'
RESULT_FIG_FILE = 'result.png'

BATCH_SIZE = 262144

# ============================================================================
# 1. LOAD THE WHOLE-IMAGE TABLE
# (the trained model is rebuilt by load_model in lib/mlp_util.py)
# ============================================================================
def load_table(table_path: Path, checkpoint: dict):
    """
    Returns X (features scaled exactly like in training), y (0 = background,
    1 = soil), the table itself (for Row / Col / R / G / B) and the image size.
    The table must contain every pixel of the image exactly once.
    """
    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            "Run 01_image_to_table.py first."
        )
    df = pd.read_csv(table_path)
    print(f"[Data] Table: {table_path.name} | shape: {df.shape}")

    # Whole-image check: the table size gives the image size, and every (Row, Col) must be present once
    rows, cols = df['Row'].values, df['Col'].values
    H, W = int(rows.max()) + 1, int(cols.max()) + 1
    seen = np.zeros((H, W), dtype=np.int32)
    np.add.at(seen, (rows, cols), 1)
    if not (seen == 1).all():
        raise ValueError(
            f"{table_path.name} does not cover the whole {H} x {W} image exactly once "
            f"({int((seen == 0).sum()):,} pixels missing, {int((seen > 1).sum()):,} repeated). "
            "This script deploys on a whole-image table such as 01_image_to_table/test.csv."
        )
    print(f"[Data] Whole image: {H} x {W} = {H * W:,} pixels, each exactly once")

    background_name = checkpoint['class_names'][0]
    X_numpy = df[checkpoint['feature_cols']].values.astype(np.float32) / checkpoint['feature_scale']
    y_numpy = (df['Class'] != background_name).values.astype(np.int64)

    print(f"[Data] Class counts: {background_name}={int((y_numpy == 0).sum()):,} | "
          f"{checkpoint['class_names'][1]}={int((y_numpy == 1).sum()):,}")
    return X_numpy, y_numpy, df, (H, W)

# ============================================================================
# 2. PREDICT
# ============================================================================
def predict(model, X_numpy: np.ndarray, device: str):
    """Returns the predicted class for every row."""
    classes = []
    with torch.no_grad():
        for start in range(0, len(X_numpy), BATCH_SIZE):
            chunk = torch.from_numpy(X_numpy[start:start + BATCH_SIZE]).to(device)
            classes.append(torch.argmax(model(chunk), dim=1).cpu())
    return torch.cat(classes).numpy()

# ============================================================================
# 3. METRICS (summarize is in lib/mlp_util.py)
# ============================================================================
def confusion(preds: np.ndarray, targets: np.ndarray):
    """Confusion counts for the soil class (1)."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    return tp, fp, fn, tn

# ============================================================================
# 4. MAIN
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained soil MLP on every pixel of the test image and save the "
                    "full-size predicted mask, a result figure and the test metrics."
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--table", type=str, default=str(DEFAULT_TABLE),
                        help="Whole-image (Row, Col, R, G, B, Class) table")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on: {device}")

    # 1) Load the trained model (no training here)
    model, checkpoint = load_model(Path(args.model), device)
    background_name, soil_name = checkpoint['class_names']

    # 2) Load the whole-image table of the test image (never seen in training or validation)
    X_numpy, y_numpy, df, (H, W) = load_table(Path(args.table), checkpoint)
    rows, cols = df['Row'].values, df['Col'].values

    # 3) Predict every pixel
    preds = predict(model, X_numpy, device)

    # 4) Test report
    tp, fp, fn, tn = confusion(preds, y_numpy)
    acc, precision, recall, f1, iou = summarize(tp, fp, fn, tn)
    print("\n--- Whole-image Test Report ---")
    print("Confusion (rows = true, cols = predicted):")
    print(f"  {background_name:>12s}: {tn:>9,d} {fp:>9,d}")
    print(f"  {soil_name:>12s}: {fn:>9,d} {tp:>9,d}")
    print(f"Accuracy: {acc:.2f}% | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

    # 5) Put the predictions back on the image grid with (Row, Col)
    pred_grid = np.zeros((H, W), dtype=np.int64)
    pred_grid[rows, cols] = preds
    true_grid = np.zeros((H, W), dtype=np.int64)
    true_grid[rows, cols] = y_numpy

    # Full-size mask: every pixel is predicted, so there is no "not used" value
    pred_vis = np.where(pred_grid == 1, SOIL_VALUE, BACKGROUND_VALUE).astype(np.uint8)
    mask_path = output_dir / PRED_MASK_FILE
    Image.fromarray(pred_vis).save(mask_path)
    print(f"\n[ok] Prediction mask saved to {mask_path} "
          f"({SOIL_VALUE} = soil, {BACKGROUND_VALUE} = background, {H} x {W})")

    # 6) Figure: image | ground truth | prediction | errors (whole image, no cropping)
    # Grey = background, white = soil (same in every panel).
    # Errors panel starts from the prediction; red = false positive, blue = false negative.
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[rows, cols] = df[['R', 'G', 'B']].values.astype(np.uint8)       # rebuilt from the table

    def class_canvas(grid):
        canvas = np.empty((H, W, 3), dtype=np.uint8)
        canvas[:] = COLOR_BACKGROUND
        canvas[grid == 1] = COLOR_SOIL
        return canvas

    errors_canvas = class_canvas(pred_grid)
    errors_canvas[(pred_grid == 1) & (true_grid == 0)] = COLOR_FALSE_POSITIVE
    errors_canvas[(pred_grid == 0) & (true_grid == 1)] = COLOR_FALSE_NEGATIVE

    panels = [
        ("Test image", image),
        ("Ground truth", class_canvas(true_grid)),
        ("Prediction", class_canvas(pred_grid)),
        ("Errors", errors_canvas),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 8.5))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    legend_items = [
        Patch(facecolor=np.array(color) / 255, edgecolor="0.3", label=label)
        for label, color in [
            ("Background", COLOR_BACKGROUND),
            ("Soil", COLOR_SOIL),
            ("False positive", COLOR_FALSE_POSITIVE),
            ("False negative", COLOR_FALSE_NEGATIVE),
        ]
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    fig_path = output_dir / RESULT_FIG_FILE
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Result figure saved to {fig_path}")

if __name__ == "__main__":
    main()
