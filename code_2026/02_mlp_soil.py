import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from matplotlib.patches import Patch
from torch.utils.data import TensorDataset, DataLoader
from typing import Iterable, Callable

# ============================================================================
# Shared global plot style (main_code/lib/control_plot.py)
# ============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))

from control_plot import (
    PLOT_PARAMS, BACKGROUND_VALUE, SOIL_VALUE, COLOR_BACKGROUND, COLOR_SOIL,
)

plt.rcParams.update(PLOT_PARAMS)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Paths are relative to this script, so it runs from any folder.
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_PATH = SCRIPT_DIR / 'soil' / '34696_0_img.jpg'
TRAIN_VAL_TABLE = SCRIPT_DIR / '01_image_to_table' / 'train_val.csv'   # made by 01_image_to_table.py

# Output folder (one folder per script, named after the script)
OUTDIR = SCRIPT_DIR / '02_mlp_soil'
OUTDIR.mkdir(parents=True, exist_ok=True)

PRED_MASK_FILE = 'predicted_mask.png'
RESULT_FIG_FILE = 'result.png'
MODEL_WEIGHT_FILE = 'mlp_model.pth'

NUM_CLASSES = 2                    # Binary: 0 (background), 1 (soil)
FEATURE_COLS = ['R', 'G', 'B']     # 3 features: pixel color
FEATURE_SCALE = 255.0              # features are divided by this (saved with the model)
BACKGROUND_NAME = 'Background'     # every other class name in the table is soil

SEED = 0
BATCH_SIZE = 2048
EPOCHS = 20
LR = 0.001
HIDDEN = [128, 64, 32]
DROPOUT = 0.1
USE_BATCHNORM = True

# ============================================================================
# 1. MLP ARCHITECTURE
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
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for ReLU-based MLP
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================================
# 2. LOAD THE TRAIN/VAL TABLE
# ============================================================================
def load_table(table_path: Path):
    """
    1) Reads train_val.csv made by 01_image_to_table.py (circles only)
    2) Returns X (R, G, B scaled to 0..1), y (0 = background, 1 = soil),
       each pixel's (Row, Col), and the soil class name
    """
    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            "Run 01_image_to_table.py first."
        )
    df = pd.read_csv(table_path)
    print(f"[Data] Table: {table_path.name} | shape: {df.shape}")

    soil_names = [c for c in df['Class'].unique() if c != BACKGROUND_NAME]
    soil_name = soil_names[0] if soil_names else 'Soil'

    X_numpy = df[FEATURE_COLS].values.astype(np.float32) / FEATURE_SCALE         # (N, 3)
    y_numpy = (df['Class'] != BACKGROUND_NAME).values.astype(np.int64)           # (N,)

    print(f"[Data] Class counts: {BACKGROUND_NAME}={int((y_numpy == 0).sum()):,} | "
          f"{soil_name}={int((y_numpy == 1).sum()):,}")
    return X_numpy, y_numpy, df['Row'].values, df['Col'].values, soil_name

# ============================================================================
# 3. VALIDATION METRICS
# ============================================================================
def evaluate(model, loader, device):
    """Confusion counts for the soil class (1) over a loader."""
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = torch.argmax(model(X_b), dim=1)
            tp += ((preds == 1) & (y_b == 1)).sum().item()
            fp += ((preds == 1) & (y_b == 0)).sum().item()
            fn += ((preds == 0) & (y_b == 1)).sum().item()
            tn += ((preds == 0) & (y_b == 0)).sum().item()
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
# 4. MAIN TRAINING + PREDICTION PIPELINE
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) Load tabular data (circles only; the test blobs are never loaded here)
    X_numpy, y_numpy, rows, cols, soil_name = load_table(TRAIN_VAL_TABLE)

    # 2) Convert to torch tensors
    X_tensor = torch.from_numpy(X_numpy)  # (N, D)
    y_tensor = torch.from_numpy(y_numpy)  # (N,)

    # Dataset and split
    dataset = TensorDataset(X_tensor, y_tensor)
    n_total = len(dataset)
    train_size = int(0.8 * n_total)
    val_size = n_total - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # drop_last avoids a 1-sample final batch, which BatchNorm cannot train on
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[Info] Input Features: {X_tensor.shape[1]} | Target Classes: {NUM_CLASSES}")
    print(f"[Info] Training samples: {train_size} | Validation samples: {val_size}")

    # 3) Init MLP
    model = MLP(
        in_dim=X_tensor.shape[1],
        hidden=HIDDEN,
        out_dim=NUM_CLASSES,
        batchnorm=USE_BATCHNORM,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4) Training loop
    print("\n--- Training ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_acc, _, _, _, val_iou = summarize(*evaluate(model, val_loader, device))
        avg_loss = train_loss / max(1, len(train_loader))

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val IoU ({soil_name}): {val_iou:.4f}")

    # Final validation report (accuracy alone hides errors on the smaller class)
    tp, fp, fn, tn = evaluate(model, val_loader, device)
    acc, precision, recall, f1, iou = summarize(tp, fp, fn, tn)
    print("\n--- Validation Report ---")
    print(f"Confusion (rows = true, cols = predicted):")
    print(f"  {BACKGROUND_NAME:>12s}: {tn:>9,d} {fp:>9,d}")
    print(f"  {soil_name:>12s}: {fn:>9,d} {tp:>9,d}")
    print(f"Accuracy: {acc:.2f}% | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

    # 5) Prediction map over the train/val pixels, placed back on the image with (Row, Col)
    print("\n--- Generating Prediction Map ---")
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, n_total, 262144):
            chunk = X_tensor[start:start + 262144].to(device)
            chunks.append(torch.argmax(model(chunk), dim=1).cpu())
    preds = torch.cat(chunks).numpy()  # (N,)

    W, H = Image.open(IMG_PATH).size
    pred_vis = np.full((H, W), BACKGROUND_VALUE, dtype=np.uint8)
    pred_vis[rows[preds == 1], cols[preds == 1]] = SOIL_VALUE

    out_path = OUTDIR / PRED_MASK_FILE
    Image.fromarray(pred_vis).save(out_path)
    print(f"[ok] Prediction mask saved to {out_path} "
          f"({SOIL_VALUE} = soil, {BACKGROUND_VALUE} = background / not in train_val.csv)")
    print(f"[ok] Predicted mask shape: {pred_vis.shape}")

    # Image | ground truth | prediction
    # Grey = background (and any pixel not in train_val.csv, e.g. the test blobs), white = soil
    image = np.array(Image.open(IMG_PATH).convert("RGB"))

    def class_canvas(values):
        canvas = np.full((H, W, 3), COLOR_BACKGROUND, dtype=np.uint8)
        canvas[rows[values == 1], cols[values == 1]] = COLOR_SOIL
        return canvas

    panels = [
        ("Image", image),
        ("Ground truth (train/val)", class_canvas(y_numpy)),
        ("Prediction (train/val)", class_canvas(preds)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.5))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    legend_items = [
        Patch(facecolor=np.array(color) / 255, edgecolor="0.3", label=label)
        for label, color in [
            ("Background / not used", COLOR_BACKGROUND),
            ("Soil", COLOR_SOIL),
        ]
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    fig_path = OUTDIR / RESULT_FIG_FILE
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Result figure saved to {fig_path}")

    # 6) Save model weights (everything 03_deploy_soil.py needs to rebuild and run it)
    model_path = OUTDIR / MODEL_WEIGHT_FILE
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_tensor.shape[1],
        'feature_cols': FEATURE_COLS,
        'feature_scale': FEATURE_SCALE,
        'class_names': [BACKGROUND_NAME, soil_name],
        'hidden': HIDDEN,
        'num_classes': NUM_CLASSES,
        'use_batchnorm': USE_BATCHNORM,
        'dropout': DROPOUT,
        'image_shape': (H, W)
    }, model_path)
    print(f"[ok] Model weights saved to {model_path}")

if __name__ == "__main__":
    main()
