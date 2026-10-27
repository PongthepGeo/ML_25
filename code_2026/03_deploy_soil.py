import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from matplotlib.patches import Patch
from typing import Iterable, Callable

# ============================================================================
# Shared global plot style (main_code/lib/control_plot.py)
# ============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))

from control_plot import (
    PLOT_PARAMS, BACKGROUND_VALUE, SOIL_VALUE, COLOR_BACKGROUND, COLOR_SOIL,
    COLOR_FALSE_POSITIVE, COLOR_FALSE_NEGATIVE,
)

plt.rcParams.update(PLOT_PARAMS)

# ============================================================================
# DEPLOYMENT: load the model trained by 02_mlp_soil.py and run it on test.csv
# No training happens here.
# ============================================================================
# Paths are relative to this script, so it runs from any folder.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / 'soil' / '34696_0_img.jpg'
DEFAULT_MODEL = SCRIPT_DIR / '02_mlp_soil' / 'mlp_model.pth'              # made by 02_mlp_soil.py
DEFAULT_TABLE = SCRIPT_DIR / '01_image_to_table' / 'test.csv'             # made by 01_image_to_table.py
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / '03_deploy_soil'

PRED_TABLE_FILE = 'test_predictions.csv'
PRED_MASK_FILE = 'predicted_mask.png'
RESULT_FIG_FILE = 'result.png'

BATCH_SIZE = 262144

# ============================================================================
# 1. MLP ARCHITECTURE (same as 02_mlp_soil.py, needed to rebuild the model)
# ============================================================================
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Iterable[int],
        out_dim: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # Final output layer
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================================
def load_model(model_path: Path, device: str):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run 02_mlp_soil.py first."
        )
    checkpoint = torch.load(model_path, map_location=device)

    model = MLP(
        in_dim=checkpoint['input_dim'],
        hidden=checkpoint['hidden'],
        out_dim=checkpoint['num_classes'],
        batchnorm=checkpoint['use_batchnorm'],
        dropout=checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()                   # inference mode: BatchNorm uses saved statistics, Dropout is off

    print(f"[Model] Loaded: {model_path}")
    print(f"[Model] Features: {checkpoint['feature_cols']} | Hidden: {checkpoint['hidden']} | "
          f"Classes: {checkpoint['class_names']}")
    return model, checkpoint

# ============================================================================
# 3. LOAD THE TEST TABLE
# ============================================================================
def load_table(table_path: Path, checkpoint: dict):
    """
    Returns X (features scaled exactly like in training), y (0 = background,
    1 = soil), and the table itself (for Row / Col).
    """
    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            "Run 01_image_to_table.py first."
        )
    df = pd.read_csv(table_path)
    print(f"[Data] Table: {table_path.name} | shape: {df.shape}")

    background_name = checkpoint['class_names'][0]
    X_numpy = df[checkpoint['feature_cols']].values.astype(np.float32) / checkpoint['feature_scale']
    y_numpy = (df['Class'] != background_name).values.astype(np.int64)

    print(f"[Data] Class counts: {background_name}={int((y_numpy == 0).sum()):,} | "
          f"{checkpoint['class_names'][1]}={int((y_numpy == 1).sum()):,}")
    return X_numpy, y_numpy, df

# ============================================================================
# 4. PREDICT
# ============================================================================
def predict(model, X_numpy: np.ndarray, device: str):
    """Returns the predicted class and the soil probability for every row."""
    classes, soil_probs = [], []
    with torch.no_grad():
        for start in range(0, len(X_numpy), BATCH_SIZE):
            chunk = torch.from_numpy(X_numpy[start:start + BATCH_SIZE]).to(device)
            probs = torch.softmax(model(chunk), dim=1)
            classes.append(torch.argmax(probs, dim=1).cpu())
            soil_probs.append(probs[:, 1].cpu())
    return torch.cat(classes).numpy(), torch.cat(soil_probs).numpy()

# ============================================================================
# 5. METRICS
# ============================================================================
def confusion(preds: np.ndarray, targets: np.ndarray):
    """Confusion counts for the soil class (1)."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    return tp, fp, fn, tn

def summarize(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = 100.0 * (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return acc, precision, recall, f1, iou

# ============================================================================
# 6. MAIN
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained soil MLP on a (Row, Col, R, G, B, Class) table and report test metrics."
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--table", type=str, default=str(DEFAULT_TABLE))
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
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

    # 2) Load the test table (blobs only, never seen in training)
    X_numpy, y_numpy, df = load_table(Path(args.table), checkpoint)

    # 3) Predict
    preds, soil_probs = predict(model, X_numpy, device)

    # 4) Test report
    tp, fp, fn, tn = confusion(preds, y_numpy)
    acc, precision, recall, f1, iou = summarize(tp, fp, fn, tn)
    print("\n--- Test Report ---")
    print("Confusion (rows = true, cols = predicted):")
    print(f"  {background_name:>12s}: {tn:>9,d} {fp:>9,d}")
    print(f"  {soil_name:>12s}: {fn:>9,d} {tp:>9,d}")
    print(f"Accuracy: {acc:.2f}% | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

    # 5) Save predictions as a table
    names = np.array(checkpoint['class_names'])
    df_pred = df[['Row', 'Col', 'Class']].copy()
    df_pred['Predicted'] = names[preds]
    df_pred['P_soil'] = soil_probs.round(4)
    table_path = output_dir / PRED_TABLE_FILE
    df_pred.to_csv(table_path, index=False)
    print(f"\n[ok] Predictions saved to {table_path}")

    # 6) Place predictions back on the image with (Row, Col)
    H, W = checkpoint['image_shape']
    rows, cols = df['Row'].values, df['Col'].values
    pred_vis = np.full((H, W), BACKGROUND_VALUE, dtype=np.uint8)
    pred_vis[rows[preds == 1], cols[preds == 1]] = SOIL_VALUE
    mask_path = output_dir / PRED_MASK_FILE
    Image.fromarray(pred_vis).save(mask_path)
    print(f"[ok] Prediction mask saved to {mask_path} "
          f"({SOIL_VALUE} = soil, {BACKGROUND_VALUE} = background / not in the test table)")

    # Zoomed figure: image | ground truth | prediction | errors
    # Grey = background (and any pixel not in the test table), white = soil (same in every panel).
    # Errors panel starts from the prediction; red = false positive, blue = false negative.
    y0, y1 = rows.min(), rows.max() + 1
    x0, x1 = cols.min(), cols.max() + 1
    image = np.array(Image.open(args.image).convert("RGB"))[y0:y1, x0:x1]

    def class_canvas(values):
        canvas = np.full((H, W, 3), COLOR_BACKGROUND, dtype=np.uint8)
        canvas[rows[values == 1], cols[values == 1]] = COLOR_SOIL
        return canvas

    errors_canvas = class_canvas(preds)
    false_positive = (preds == 1) & (y_numpy == 0)
    false_negative = (preds == 0) & (y_numpy == 1)
    errors_canvas[rows[false_positive], cols[false_positive]] = COLOR_FALSE_POSITIVE
    errors_canvas[rows[false_negative], cols[false_negative]] = COLOR_FALSE_NEGATIVE

    panels = [
        ("Test image", image),
        ("Ground truth", class_canvas(y_numpy)[y0:y1, x0:x1]),
        ("Prediction", class_canvas(preds)[y0:y1, x0:x1]),
        ("Errors", errors_canvas[y0:y1, x0:x1]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    legend_items = [
        Patch(facecolor=np.array(color) / 255, edgecolor="0.3", label=label)
        for label, color in [
            ("Background / not used", COLOR_BACKGROUND),
            ("Soil", COLOR_SOIL),
            ("False positive", COLOR_FALSE_POSITIVE),
            ("False negative", COLOR_FALSE_NEGATIVE),
        ]
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    fig_path = output_dir / RESULT_FIG_FILE
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Result figure saved to {fig_path}")

if __name__ == "__main__":
    main()
