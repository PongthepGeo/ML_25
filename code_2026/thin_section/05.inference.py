import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =========================================================
# Shared global plot style (main_code/lib/control_plot.py)
# =========================================================
sys.path.append(str(Path(__file__).resolve().parents[2] / "lib"))

import matplotlib.pyplot as plt
from control_plot import PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)

# =========================================================
# Paths
# =========================================================
model_path = Path("04_xgboost_model.json")
image_path = Path("data/img_0145.png")
label_path = Path("data/label_0145.png")
output_dir = Path("05_inference")
output_figure = output_dir / "inference_results.png"

# =========================================================
# Reference label colors (same mapping used in 02/03)
# =========================================================
reference_colors = {
    "Grey": np.array([128, 128, 128]),
    "Blue": np.array([0, 0, 255]),
    "Green": np.array([0, 255, 0]),
    "Yellow": np.array([255, 255, 0]),
}

# Class order matches sklearn LabelEncoder's alphabetical sort,
# which is what 04.xgboost.py used to encode "Class" before training.
class_names = sorted(reference_colors.keys())

# =========================================================
# Find nearest label class
# =========================================================
def nearest_color_name(rgb):
    rgb = np.asarray(rgb, dtype=float)
    distances = {
        name: np.linalg.norm(rgb - ref.astype(float))
        for name, ref in reference_colors.items()
    }
    return min(distances, key=distances.get)

# =========================================================
# Load trained model
# =========================================================
print("\n====================================")
print("Load trained model")
print("====================================")

model = XGBClassifier()
model.load_model(model_path)
print(f"Loaded model : {model_path}")
print(f"Classes ({len(class_names)}) : {class_names}")

# =========================================================
# Load held-out test image and ground-truth label
# =========================================================
print("\n====================================")
print("Load test image")
print("====================================")

image = np.array(Image.open(image_path).convert("RGB"))
label = np.array(Image.open(label_path).convert("RGB"))

if image.shape[:2] != label.shape[:2]:
    raise ValueError(
        "Image and label dimensions do not match.\n"
        f"Image: {image.shape}\n"
        f"Label: {label.shape}"
    )

height, width = image.shape[:2]
print(f"Image : {image_path} {image.shape}")
print(f"Label : {label_path} {label.shape}")

# =========================================================
# Ground-truth class map (from label colors)
# =========================================================
print("\nDetected label colors:")

detected_colors, counts = np.unique(label.reshape(-1, 3), axis=0, return_counts=True)

true_id_map = np.full((height, width), -1, dtype=np.int64)

for rgb, count in zip(detected_colors, counts):
    class_name = nearest_color_name(rgb)
    class_id = class_names.index(class_name)
    mask = np.all(label == rgb, axis=2)
    true_id_map[mask] = class_id
    print(f"{class_name:8s} | Label RGB={tuple(rgb)} | Pixels={count:,}")

y_true = true_id_map.flatten()

# =========================================================
# Predict on every pixel
# =========================================================
print("\n====================================")
print("Prediction")
print("====================================")

X_infer = image.reshape(-1, 3).astype(np.float64)
print(f"Pixels to classify : {len(X_infer):,}")

predict_start = time.time()
y_pred = model.predict(X_infer)
predict_time = time.time() - predict_start
print(f"Prediction time : {predict_time:.2f} seconds")

pred_id_map = y_pred.reshape(height, width)

# =========================================================
# Accuracy
# =========================================================
accuracy = accuracy_score(y_true, y_pred)
print("\n====================================")
print("Inference Results")
print("====================================")
print(f"Accuracy: {accuracy:.4f}")

# =========================================================
# Classification report
# =========================================================
print("\n====================================")
print("Classification Report")
print("====================================\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# =========================================================
# Confusion matrix
# =========================================================
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{name}" for name in class_names],
    columns=[f"Pred_{name}" for name in class_names],
)

print("\n====================================")
print("Confusion Matrix")
print("====================================\n")
print(cm_df)

# =========================================================
# Build RGB visualizations from class-id maps
# =========================================================
def id_map_to_rgb(id_map):
    rgb_image = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    for class_id, name in enumerate(class_names):
        rgb_image[id_map == class_id] = reference_colors[name]
    return rgb_image

true_rgb = id_map_to_rgb(true_id_map)
pred_rgb = id_map_to_rgb(pred_id_map)
diff_mask = (true_id_map != pred_id_map)

print("\n====================================")
print("Difference Summary")
print("====================================")
print(f"Correct pixels   : {(~diff_mask).sum():,}")
print(f"Incorrect pixels : {diff_mask.sum():,}")
print(f"Total pixels     : {diff_mask.size:,}")

# =========================================================
# Figure: Image | True Label | Predicted Label | Difference
# =========================================================
# four image panels -- intentionally not the global 16:9 default
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image)
axes[0].set_title("Image")
axes[0].axis("off")

axes[1].imshow(true_rgb)
axes[1].set_title("True Label")
axes[1].axis("off")

axes[2].imshow(pred_rgb)
axes[2].set_title(f"XGBoost Prediction\n(Accuracy: {accuracy:.4f})")
axes[2].axis("off")

axes[3].imshow(diff_mask, cmap="Reds")
axes[3].set_title(f"Difference\n({diff_mask.sum():,} px)")
axes[3].axis("off")

plt.tight_layout()

output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_figure, format="png", bbox_inches="tight")
print(f"\nSaved figure: {output_figure}")

plt.show()

print("\nDone.")
