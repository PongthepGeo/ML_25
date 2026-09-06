import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Iterable, Callable

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = '16_mlp/mlp_model.pth'  # weights written by 16_mlp.py
TEST_IMG_PATH = 'midterm_xgboost/data/img.png'  # Path to test image

# Output folder (one folder per script, named after the script)
OUTDIR = Path('17_inferrence')
OUTDIR.mkdir(parents=True, exist_ok=True)

INFERENCE_MASK_FILE = 'inference_mask.png'

# ============================================================================
# 1. MLP ARCHITECTURE (Same as training)
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
# 2. LOAD MODEL AND METADATA
# ============================================================================
def load_model(model_path: str, device: str):
    """
    Loads the saved model checkpoint and reconstructs the MLP model.

    Returns:
        model: The loaded MLP model
        metadata: Dictionary containing model configuration
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[Info] Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    input_dim = checkpoint['input_dim']
    hidden = checkpoint['hidden']
    num_classes = checkpoint['num_classes']
    use_batchnorm = checkpoint['use_batchnorm']
    dropout = checkpoint['dropout']
    image_shape = checkpoint.get('image_shape', None)

    # Reconstruct model
    model = MLP(
        in_dim=input_dim,
        hidden=hidden,
        out_dim=num_classes,
        batchnorm=use_batchnorm,
        dropout=dropout
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[Info] Model loaded successfully")
    print(f"[Info] Input dim: {input_dim}, Hidden: {hidden}, Classes: {num_classes}")
    print(f"[Info] Image shape: {image_shape}")

    metadata = {
        'input_dim': input_dim,
        'hidden': hidden,
        'num_classes': num_classes,
        'use_batchnorm': use_batchnorm,
        'dropout': dropout,
        'image_shape': image_shape
    }

    return model, metadata

# ============================================================================
# 3. PREPROCESS TEST IMAGE
# ============================================================================
def preprocess_image(img_path: str):
    """
    Preprocesses the test image in the same way as training data.

    Returns:
        X_tensor: Preprocessed features as torch tensor
        H, W: Original image dimensions
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    H, W, _ = img.shape
    print(f"[Data] Test image shape: {img.shape}")

    # Extract first channel and normalize (same as training)
    img_channel = img[:, :, 0].astype(np.float32)
    img_vector = img_channel.flatten() / 255.0

    # Convert to tensor
    X_tensor = torch.from_numpy(img_vector).unsqueeze(1)  # (N, 1)

    return X_tensor, H, W

# ============================================================================
# 4. INFERENCE PIPELINE
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running inference on: {device}\n")

    # 1) Load trained model
    model, metadata = load_model(MODEL_PATH, device)

    # 2) Load and preprocess test image
    print(f"\n[Info] Loading test image from {TEST_IMG_PATH}")
    X_tensor, H, W = preprocess_image(TEST_IMG_PATH)

    # Validate dimensions
    expected_shape = metadata['image_shape']
    if expected_shape and (H, W) != expected_shape:
        print(f"[Warning] Test image shape {(H, W)} differs from training shape {expected_shape}")

    # 3) Run inference
    print("\n[Info] Running inference...")
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        logits = model(X_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    # 4) Reshape to image dimensions
    pred_map = predictions.reshape(H, W).astype(np.uint8)

    # Calculate class distribution
    unique, counts = np.unique(pred_map, return_counts=True)
    print(f"\n[Results] Class distribution:")
    for cls, count in zip(unique, counts):
        percentage = 100.0 * count / pred_map.size
        print(f"  Class {cls}: {count} pixels ({percentage:.2f}%)")

    # 5) Save prediction mask
    # Map 0->0 (black), 1->255 (white) for visualization
    pred_vis = (pred_map * 255).astype(np.uint8)

    out_path = OUTDIR / INFERENCE_MASK_FILE
    cv2.imwrite(str(out_path), pred_vis)

    print(f"\n[ok] Inference mask saved to {out_path}")
    print(f"[ok] Prediction shape: {pred_map.shape}")

    # Optional: Save colored overlay
    original_img = cv2.imread(TEST_IMG_PATH, cv2.IMREAD_COLOR)
    if original_img is not None:
        overlay = original_img.copy()
        # Highlight class 1 pixels in green
        mask = pred_map == 1
        # Blend original pixels with green color
        green_color = np.array([0, 255, 0], dtype=np.uint8)
        overlay[mask] = (original_img[mask] * 0.6 + green_color * 0.4).astype(np.uint8)

        overlay_path = OUTDIR / 'inference_overlay.png'
        cv2.imwrite(str(overlay_path), overlay)
        print(f"[ok] Overlay image saved to {overlay_path}")

if __name__ == "__main__":
    main()
