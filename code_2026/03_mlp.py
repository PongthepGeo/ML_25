import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# Plot style: control_plot.py in THIS folder (code_2026/control_plot.py).
# Model, metrics and checkpoint format: main_code/lib/mlp_util.py (shared with 04_deploy.py).
# ============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))                  # local control_plot.py always wins
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))         # mlp_util.py

from control_plot import PLOT_PARAMS
from mlp_util import MLP, summarize, save_checkpoint

plt.rcParams.update(PLOT_PARAMS)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Paths are relative to this script, so it runs from any folder.
SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR / '02_split'                    # made by 02_split.py (split of image 1)
TRAIN_TABLE = SPLIT_DIR / 'train.csv'
VAL_TABLE = SPLIT_DIR / 'val.csv'
# Test = a different photo (image 2), made by 01_image_to_table.py; never seen in training
TEST_TABLE = SCRIPT_DIR / '01_image_to_table' / 'test.csv'

# Output folder (one folder per script, named after the script)
OUTDIR = SCRIPT_DIR / '03_mlp'
OUTDIR.mkdir(parents=True, exist_ok=True)

LOSS_HISTORY_FILE = 'loss_history.png'
CONFUSION_FIG_FILE = 'confusion_matrices.png'
MODEL_WEIGHT_FILE = 'mlp_model.pth'

NUM_CLASSES = 2                    # Binary: 0 (background), 1 (soil)
FEATURE_COLS = ['R', 'G', 'B']     # 3 features: pixel color
FEATURE_SCALE = 255.0              # features are divided by this (saved with the model)
BACKGROUND_NAME = 'Background'     # every other class name in the table is soil
CLASS_LABELS = ['Background', 'Soil']   # axis labels of the confusion matrices

SEED = 0
BATCH_SIZE = 2048
EPOCHS = 20
LR = 0.001
HIDDEN = [128, 64, 32]
DROPOUT = 0.1
USE_BATCHNORM = True

# ============================================================================
# 1. MLP ARCHITECTURE: class MLP in lib/mlp_util.py (shared with 04_deploy.py)
# ============================================================================

# ============================================================================
# 2. LOAD A TABLE (train / val / test)
# ============================================================================
def load_table(table_path: Path):
    """
    1) Reads one table (train.csv / val.csv from 02_split.py, test.csv from 01_image_to_table.py)
    2) Returns X (R, G, B scaled to 0..1), y (0 = background, 1 = soil),
       and the soil class name
    """
    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            "Run 01_image_to_table.py and 02_split.py first."
        )
    df = pd.read_csv(table_path)

    soil_names = [c for c in df['Class'].unique() if c != BACKGROUND_NAME]
    soil_name = soil_names[0] if soil_names else 'Soil'

    X_numpy = df[FEATURE_COLS].values.astype(np.float32) / FEATURE_SCALE         # (N, 3)
    y_numpy = (df['Class'] != BACKGROUND_NAME).values.astype(np.int64)           # (N,)

    print(f"[Data] {table_path.name:10s} | rows: {len(df):>9,} | "
          f"{BACKGROUND_NAME}={int((y_numpy == 0).sum()):,} | {soil_name}={int((y_numpy == 1).sum()):,}")
    return X_numpy, y_numpy, soil_name

def make_loader(X_numpy, y_numpy, shuffle, drop_last=False):
    dataset = TensorDataset(torch.from_numpy(X_numpy), torch.from_numpy(y_numpy))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=drop_last)

# ============================================================================
# 3. METRICS
# ============================================================================
def evaluate(model, loader, criterion, device):
    """Average loss and confusion counts for the soil class (1) over a loader."""
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss_sum += criterion(logits, y_b).item() * len(y_b)
            n_samples += len(y_b)
            preds = torch.argmax(logits, dim=1)
            tp += ((preds == 1) & (y_b == 1)).sum().item()
            fp += ((preds == 1) & (y_b == 0)).sum().item()
            fn += ((preds == 0) & (y_b == 1)).sum().item()
            tn += ((preds == 0) & (y_b == 0)).sum().item()
    return loss_sum / max(1, n_samples), (tp, fp, fn, tn)

# summarize(tp, fp, fn, tn) -> accuracy, precision, recall, F1, IoU: see lib/mlp_util.py

def print_report(title, counts, soil_name):
    tp, fp, fn, tn = counts
    acc, precision, recall, f1, iou = summarize(tp, fp, fn, tn)
    print(f"\n--- {title} ---")
    print("Confusion (rows = true, cols = predicted):")
    print(f"  {BACKGROUND_NAME:>12s}: {tn:>9,d} {fp:>9,d}")
    print(f"  {soil_name:>12s}: {fn:>9,d} {tp:>9,d}")
    print(f"Accuracy: {acc:.2f}% | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

# ============================================================================
# 4. PLOTS: LOSS HISTORY AND CONFUSION MATRICES
# ============================================================================
def plot_loss_history(history, fig_path):
    epochs = np.arange(1, len(history['train_loss']) + 1)
    best = int(np.argmin(history['val_loss']))

    fig, ax = plt.subplots()
    ax.plot(epochs, history['train_loss'], color='tab:blue', linewidth=2, marker='o',
            label='Train loss')
    ax.plot(epochs, history['val_loss'], color='tab:orange', linewidth=2, marker='s',
            label='Val loss')
    ax.plot(epochs[best], history['val_loss'][best], '*', color='black', markersize=18,
            label=f'Lowest val loss (epoch {epochs[best]})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('Loss History')
    ax.set_xticks(epochs[::2])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"[ok] Loss history saved to {fig_path}")
    return fig

def plot_confusion_matrices(val_counts, test_counts, fig_path):
    """Validation and test confusion matrices side by side.
    Each cell shows the pixel count and its share of the true class (row)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (title, (tp, fp, fn, tn)) in zip(axes, [('Validation', val_counts), ('Test', test_counts)]):
        matrix = np.array([[tn, fp], [fn, tp]])                 # rows = true, cols = predicted
        share = matrix / matrix.sum(axis=1, keepdims=True)      # row-normalised: colors show rates
        ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}\n({100 * share[i, j]:.2f}%)",
                        ha='center', va='center', fontsize=20,
                        color='white' if share[i, j] > 0.5 else 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASS_LABELS)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASS_LABELS)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{title} (n = {matrix.sum():,})')
    fig.suptitle('Confusion Matrices (count and % of the true class)', fontsize=22, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"[ok] Confusion matrices saved to {fig_path}")
    return fig

# ============================================================================
# 5. MAIN: TRAIN (loss history) -> VAL -> TEST
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) Load the three tables: train + val (image 1, 02_split.py) and test (image 2, 01_image_to_table.py)
    X_train, y_train, soil_name = load_table(TRAIN_TABLE)
    X_val, y_val, _ = load_table(VAL_TABLE)
    X_test, y_test, _ = load_table(TEST_TABLE)

    # drop_last avoids a 1-sample final batch, which BatchNorm cannot train on
    train_loader = make_loader(X_train, y_train, shuffle=True, drop_last=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    print(f"[Info] Input Features: {len(FEATURE_COLS)} | Target Classes: {NUM_CLASSES}")
    print(f"[Info] Training samples: {len(X_train)} | Validation samples: {len(X_val)} | "
          f"Test samples: {len(X_test)}")

    # 2) Init MLP
    model = MLP(
        in_dim=len(FEATURE_COLS),
        hidden=HIDDEN,
        out_dim=NUM_CLASSES,
        batchnorm=USE_BATCHNORM,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Training loop, keeping the loss history (the test table is NOT touched here)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_iou': []}
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

        avg_loss = train_loss / max(1, len(train_loader))
        val_loss, val_counts = evaluate(model, val_loader, criterion, device)
        val_acc, _, _, _, val_iou = summarize(*val_counts)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val IoU ({soil_name}): {val_iou:.4f}")

    best_epoch = int(np.argmin(history['val_loss'])) + 1
    print(f"\n[Info] Lowest val loss {min(history['val_loss']):.4f} at epoch {best_epoch} of {EPOCHS}")

    # 4) Loss history figure (saved now, shown at the very end so nothing blocks the run)
    plot_loss_history(history, OUTDIR / LOSS_HISTORY_FILE)

    # 5) Final reports: validation vs test (the test table is evaluated once, here)
    _, val_counts = evaluate(model, val_loader, criterion, device)
    test_loss, test_counts = evaluate(model, test_loader, criterion, device)
    print_report("Validation Report", val_counts, soil_name)
    print_report("Test Report (unseen image)", test_counts, soil_name)
    print(f"Test loss: {test_loss:.4f}")

    # 6) Confusion matrices (validation and test)
    plot_confusion_matrices(val_counts, test_counts, OUTDIR / CONFUSION_FIG_FILE)

    # 7) Save model weights (+ loss history)
    model_path = OUTDIR / MODEL_WEIGHT_FILE
    save_checkpoint(
        model_path, model,
        feature_cols=FEATURE_COLS,
        feature_scale=FEATURE_SCALE,
        class_names=[BACKGROUND_NAME, soil_name],
        hidden=HIDDEN,
        num_classes=NUM_CLASSES,
        use_batchnorm=USE_BATCHNORM,
        dropout=DROPOUT,
        history=history,
    )
    print(f"[ok] Model weights saved to {model_path}")

    # 8) Show the loss history and the confusion matrices last (closing the windows ends the script)
    plt.show()

if __name__ == "__main__":
    main()
