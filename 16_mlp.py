import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Iterable, Callable

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_PATH = 'midterm_xgboost/data/img.png'
LABEL_PATH = 'midterm_xgboost/data/label.png'

TABULAR_DIR = 'tabular'
TABULAR_FILE = 'table.csv'

OUTPUT_DIR = 'figure_out/mlp_supervised'
PRED_MASK_FILE = 'predicted_mask.png'

NUM_CLASSES = 2          # Binary: 0 (background), 1 (object)
FEATURE_COLS = ['img']   # Currently 1D feature (grayscale intensity)

BATCH_SIZE = 2048
EPOCHS = 20
LR = 0.001
HIDDEN = [128, 64, 32]   # Complex MLP allowed even with simple input
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
# 2. TABULARIZATION WITH OPENCV (IMAGE + LABEL -> CSV + NUMPY)
# ============================================================================
def build_tabular_from_images(img_path: str,
                              label_path: str,
                              tabular_dir: str,
                              tabular_file: str):
    """
    1) Reads img.png and label.png with cv2
    2) Builds a tabular table (img, label)
    3) Saves table.csv
    4) Returns X (features), y (labels), and (H, W)
    """
    # --- Read images ---
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    label = cv2.imread(label_path, cv2.IMREAD_COLOR)
    if label is None:
        raise FileNotFoundError(f"Could not read label image: {label_path}")

    H, W, _ = img.shape
    print(f"[Data] Image shape: {img.shape}")
    print(f"[Data] Label shape: {label.shape}")

    # --- Feature: first channel of img (B channel), flattened ---
    # You can switch to cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if preferred.
    img_channel = img[:, :, 0].astype(np.float32)  # 0..255
    img_vector = img_channel.flatten() / 255.0     # normalize to [0,1]

    # --- Label processing: white (255) -> 0, others -> 1 ---
    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # Anything exactly 255 is treated as background (0); everything else as class 1.
    label_binary = np.where(label_gray == 255, 0, 1).astype(np.int64)
    label_vector = label_binary.flatten()

    # --- Build DataFrame ---
    data = {
        'img': img_vector,
        'label': label_vector
    }
    df = pd.DataFrame(data)

    # --- Save CSV ---
    os.makedirs(tabular_dir, exist_ok=True)
    tabular_path = os.path.join(tabular_dir, tabular_file)
    df.to_csv(tabular_path, index=False)
    print(f"[Tabular] Table saved to {tabular_path}")
    print(f"[Tabular] Shape: {df.shape}")

    # --- Return numpy arrays and spatial size ---
    X_numpy = df[FEATURE_COLS].values.astype(np.float32)  # (N, D)
    y_numpy = df['label'].values.astype(np.int64)         # (N,)

    return X_numpy, y_numpy, (H, W)

# ============================================================================
# 3. MAIN TRAINING + PREDICTION PIPELINE
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on: {device}")

    # 1) Build tabular data from images (and save CSV)
    X_numpy, y_numpy, (H, W) = build_tabular_from_images(
        IMG_PATH,
        LABEL_PATH,
        TABULAR_DIR,
        TABULAR_FILE
    )

    # 2) Convert to torch tensors
    X_tensor = torch.from_numpy(X_numpy)  # (N, D)
    y_tensor = torch.from_numpy(y_numpy)  # (N,)

    # Dataset and split
    dataset = TensorDataset(X_tensor, y_tensor)
    n_total = len(dataset)
    train_size = int(0.8 * n_total)
    val_size = n_total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
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
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)

        val_acc = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = train_loss / max(1, len(train_loader))

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 5) Full-image prediction and mask saving
    print("\n--- Generating Prediction Map ---")
    model.eval()
    with torch.no_grad():
        full_X = X_tensor.to(device)
        full_logits = model(full_X)
        full_preds = torch.argmax(full_logits, dim=1).cpu().numpy()  # (N,)

    # Reshape back to (H, W)
    pred_map = full_preds.reshape(H, W).astype(np.uint8)

    # Map 0->0, 1->255 for visualization
    pred_vis = (pred_map * 255).astype(np.uint8)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, PRED_MASK_FILE)
    cv2.imwrite(out_path, pred_vis)
    print(f"[ok] Prediction mask saved to {out_path}")
    print(f"[ok] Predicted mask shape: {pred_map.shape}")

if __name__ == "__main__":
    main()
