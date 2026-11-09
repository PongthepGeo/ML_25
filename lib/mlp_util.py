"""Shared MLP code for the soil pipeline (code_2026/03_mlp.py and 04_deploy.py).

Keeping the model definition, the metrics and the checkpoint format in one
place means the training script and the deploy script can never drift apart:
whatever 03_mlp.py saves is exactly what 04_deploy.py knows how to load.

Usage (next to the control_plot import):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))

    from mlp_util import MLP, summarize, save_checkpoint, load_model
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Iterable, Callable

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
# 2. METRICS (binary, soil = class 1)
# ============================================================================
def summarize(tp, fp, fn, tn):
    """Accuracy (%), precision, recall, F1 and IoU of the soil class from confusion counts."""
    total = tp + fp + fn + tn
    acc = 100.0 * (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return acc, precision, recall, f1, iou

# ============================================================================
# 3. SAVE / LOAD A TRAINED MODEL (the checkpoint format lives only here)
# ============================================================================
def save_checkpoint(model_path: Path, model, feature_cols, feature_scale, class_names,
                    hidden, num_classes, use_batchnorm, dropout, history):
    """Saves the weights plus everything needed to rebuild and run the model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': len(feature_cols),
        'feature_cols': feature_cols,
        'feature_scale': feature_scale,
        'class_names': class_names,
        'hidden': hidden,
        'num_classes': num_classes,
        'use_batchnorm': use_batchnorm,
        'dropout': dropout,
        'history': history
    }, model_path)

def load_model(model_path: Path, device: str):
    """Rebuilds the MLP from a checkpoint made by save_checkpoint, in inference mode."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run code_2026/03_mlp.py first."
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
