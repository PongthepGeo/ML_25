import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
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
# =========================================================
csv_path = Path("basalt.csv")
model_path = Path("04_xgboost_model.json")
output_dir = Path("06_tsne")
output_figure = output_dir / "tsne_clusters.png"

# =========================================================
# Load dataset
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
y_true = label_encoder.fit_transform(y_raw)
target_names = [f"Cluster_{c}" for c in label_encoder.classes_]
num_classes = len(target_names)

print(f"Features ({len(feature_columns)}): {feature_columns}")
print(f"Classes ({num_classes}): {target_names}")

# =========================================================
# Predicted clusters (from the trained XGBoost model)
#
# Shown side-by-side with the true labels so misclassified
# rows are visible directly in the embedding -- e.g. the
# small Cluster_4 / Cluster_7 classes that had low IoU in
# 04_xgboost.py / 05_inference.py.
# =========================================================
print("\n====================================")
print("Load trained model")
print("====================================")

model = XGBClassifier()
model.load_model(model_path)
y_pred = model.predict(X)
print(f"Loaded model : {model_path}")
print(f"Predicted samples : {len(y_pred):,}")

# =========================================================
# Standardize features
#
# t-SNE uses Euclidean distance, and these geochemical
# features span very different scales (SiO2 ~40-70 wt%,
# SR87_SR86 ~0.7, Zr up to ~2000 ppm) -- without scaling,
# the trace elements with the largest raw numbers would
# dominate the distances.
# =========================================================
X_scaled = StandardScaler().fit_transform(X)

# =========================================================
# t-SNE embedding
# =========================================================
print("\n====================================")
print("t-SNE Embedding")
print("====================================")

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
Z = tsne.fit_transform(X_scaled)
print(f"Embedding shape: {Z.shape}")

# =========================================================
# Figure: True clusters | Predicted clusters
# =========================================================
# two square scatter panels -- intentionally not the global 16:9 default
fig, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))

cmap = plt.cm.tab10

for class_id in range(num_classes):
    mask = (y_true == class_id)
    ax_true.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.8, color=cmap(class_id), label=target_names[class_id])

ax_true.set_title("t-SNE: True Cluster_ID")
ax_true.set_xlabel("t-SNE 1")
ax_true.set_ylabel("t-SNE 2")
ax_true.legend(fontsize=9, ncol=2, frameon=True)

for class_id in range(num_classes):
    mask = (y_pred == class_id)
    ax_pred.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.8, color=cmap(class_id), label=target_names[class_id])

misclassified = (y_true != y_pred)
ax_pred.scatter(Z[misclassified, 0], Z[misclassified, 1], s=60, facecolors="none",
                edgecolors="black", linewidths=1.2, label="Misclassified")

ax_pred.set_title("t-SNE: XGBoost Predicted Cluster")
ax_pred.set_xlabel("t-SNE 1")
ax_pred.set_ylabel("t-SNE 2")
ax_pred.legend(fontsize=9, ncol=2, frameon=True)

plt.tight_layout()

output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_figure, format="png", bbox_inches="tight")
print(f"\nSaved figure: {output_figure}")

plt.show()

# =========================================================
# Summary
# =========================================================
accuracy = (y_true == y_pred).mean()
print("\n====================================")
print("Summary")
print("====================================")
print(f"Overall accuracy (full dataset): {accuracy:.4f}")
print(f"Misclassified points: {int(misclassified.sum()):,} / {len(y_true):,}")

print("\nDone.")
