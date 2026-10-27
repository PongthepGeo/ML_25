import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =========================================================
# Shared global plot style (main_code/lib/control_plot.py)
# =========================================================
sys.path.append(str(Path(__file__).resolve().parents[3] / "lib"))

import matplotlib.pyplot as plt
from control_plot import PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)

# =========================================================
# Paths
#
# There is no second labeled file here (unlike the
# thin-section img_0145 test image), so "inference" means:
# reload basalt.csv, reproduce the EXACT same train/val/test
# split 04_xgboost.py used (same random_state, same
# test_size), recover the held-out test rows, and run the
# already-trained model on them.
# =========================================================
csv_path = Path("basalt.csv")
model_path = Path("04_xgboost_model.json")
output_dir = Path("05_inference")
output_figure = output_dir / "confusion_matrix.png"

# =========================================================
# Load trained model
# =========================================================
print("\n====================================")
print("Load trained model")
print("====================================")

model = XGBClassifier()
model.load_model(model_path)
print(f"Loaded model : {model_path}")

# =========================================================
# Load dataset and rebuild the label encoding
#
# LabelEncoder.fit_transform on the full column is
# deterministic (sorted unique values), so this reproduces
# the exact same Cluster_ID -> class_id mapping 04_xgboost.py
# used, without needing to persist the encoder separately.
# =========================================================
print("\n====================================")
print("Load dataset")
print("====================================")

df = pd.read_csv(csv_path)
print(f"CSV file: {csv_path}")
print(f"Dataset rows: {len(df):,}")

feature_columns = [c for c in df.columns if c not in ("Cluster_ID", "Number")]
X = df[feature_columns].values
y_raw = df["Cluster_ID"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
target_names = [f"Cluster_{c}" for c in label_encoder.classes_]

print(f"Classes ({len(target_names)}) : {target_names}")

# =========================================================
# Reproduce the held-out test split from 04_xgboost.py
# =========================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n====================================")
print("Held-Out Test Split")
print("====================================")
print(f"Testing samples : {len(X_test):,}")

print("\nTesting class counts:")
test_unique, test_counts = np.unique(y_test, return_counts=True)
for class_id, count in zip(test_unique, test_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"Cluster_ID {class_name}: " f"{count:,}")

# =========================================================
# Predict
# =========================================================
print("\n====================================")
print("Prediction")
print("====================================")

predict_start = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - predict_start
print(f"Predicted samples : {len(y_pred):,}")
print(f"Prediction time   : {predict_time:.4f} seconds")

# =========================================================
# Accuracy
# =========================================================
accuracy = accuracy_score(y_test, y_pred)
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
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# =========================================================
# Confusion matrix
# =========================================================
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{name}" for name in target_names],
    columns=[f"Pred_{name}" for name in target_names],
)

print("\n====================================")
print("Confusion Matrix")
print("====================================\n")
print(cm_df)

# =========================================================
# Per-class TP / FN / FP / TN
# =========================================================
num_classes = len(target_names)
total_samples = cm.sum()

tp_per_class = np.zeros(num_classes, dtype=np.int64)
fn_per_class = np.zeros(num_classes, dtype=np.int64)
fp_per_class = np.zeros(num_classes, dtype=np.int64)
tn_per_class = np.zeros(num_classes, dtype=np.int64)

for class_id in range(num_classes):
    true_positive = cm[class_id, class_id]
    false_positive = cm[:, class_id].sum() - true_positive
    false_negative = cm[class_id, :].sum() - true_positive
    true_negative = total_samples - true_positive - false_positive - false_negative

    tp_per_class[class_id] = true_positive
    fn_per_class[class_id] = false_negative
    fp_per_class[class_id] = false_positive
    tn_per_class[class_id] = true_negative

stats_df = pd.DataFrame(
    {
        "TP": tp_per_class,
        "FN": fn_per_class,
        "FP": fp_per_class,
        "TN": tn_per_class,
    },
    index=target_names,
)

print("\n====================================")
print("Per-Class TP / FN / FP / TN")
print("====================================\n")
print(stats_df.to_string())

# =========================================================
# Difference summary
# =========================================================
diff_mask = (y_test != y_pred)
print("\n====================================")
print("Difference Summary")
print("====================================")
print(f"Correct predictions   : {(~diff_mask).sum():,}")
print(f"Incorrect predictions : {diff_mask.sum():,}")
print(f"Total predictions     : {diff_mask.size:,}")

# =========================================================
# Figure: confusion matrix heatmap (global 16:9 default)
# =========================================================
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(target_names, rotation=45, ha="right")
ax.set_yticklabels(target_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_figure, format="png", bbox_inches="tight")
print(f"\nSaved figure: {output_figure}")

plt.show()

print("\nDone.")
