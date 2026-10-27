import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier

# =========================================================
# Paths
#
# Single CSV -- unlike the thin-section pipeline (separate
# train/test images), basalt.csv has no second labeled file,
# so train / validation / test all come from a stratified
# split of it (~64% / ~16% / 20%).
# =========================================================
csv_path = Path("basalt.csv")
model_path = Path("04_xgboost_model.json")
print("\n====================================")

# =========================================================
# Load CSV
# =========================================================
print("\n====================================")
print("Load dataset")
print("====================================")

df = pd.read_csv(csv_path)
print(f"CSV file: {csv_path}")
print(f"Dataset rows: {len(df):,}")
print(f"Dataset cols: {len(df.columns)}")
print("\nFirst 10 rows:\n")
print(df.head(10))

# Class counts
print("\n====================================")
print("Class distribution")
print("====================================")

class_counts = df["Cluster_ID"].value_counts()
print(class_counts)
print("\nClass percentages:")

class_percentages = (df["Cluster_ID"].value_counts(normalize=True).mul(100))

for class_name, percentage in class_percentages.items():
    print(f"{class_name}: " f"{percentage:.2f}%")

# =========================================================
# Features
#
# "Number" is a sample locality code (e.g. "And1", "Hwi3"),
# not a measurement -- it is almost a 1:1 proxy for
# Cluster_ID (each locality prefix maps to one cluster), so
# it is excluded to avoid leaking the label through the
# sample ID instead of the actual geochemistry.
# =========================================================
feature_columns = [c for c in df.columns if c not in ("Cluster_ID", "Number")]
X = df[feature_columns].values
y_raw = df["Cluster_ID"].values
print("\n====================================")
print("Features")
print("====================================")
print(feature_columns)
print(f"X shape: {X.shape}")

# =========================================================
# Encode classes
#
# Cluster_ID is already an integer (1-8), but XGBoost's
# multi:softprob needs contiguous 0-indexed labels, so it
# still goes through LabelEncoder for consistency with the
# rest of the pipeline.
# =========================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print("\n====================================")
print("Class encoding")
print("====================================")

for class_id, class_name in enumerate(label_encoder.classes_):
    print(f"Cluster_ID {class_name} -> {class_id}")

# =========================================================
# Train / validation / test split
#
# Step 1: hold out 20% as the FINAL test set.
#         It is never shown to the model during training or
#         monitoring below -- only used once, at the end, to
#         report unbiased metrics.
# Step 2: split the remaining 80% again (80/20) into train
#         and validation. Validation is what eval_set below
#         actually monitors during boosting.
#
# Overall split: ~64% train / ~16% validation / 20% test
# =========================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval)

print("\n====================================")
print("Train / Validation / Test Split")
print("====================================")
print(f"Training samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")
print(f"Testing samples    : {len(X_test):,}")

# =========================================================
# Check class distribution after split
# =========================================================
print("\nTraining class counts:")

train_unique, train_counts = np.unique(y_train, return_counts=True)

for class_id, count in zip(train_unique, train_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"Cluster_ID {class_name}: " f"{count:,}")

print("\nValidation class counts:")
val_unique, val_counts = np.unique(y_val, return_counts=True)

for class_id, count in zip(val_unique, val_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"Cluster_ID {class_name}: " f"{count:,}")

print("\nTesting class counts:")
test_unique, test_counts = np.unique(y_test, return_counts=True)

for class_id, count in zip(test_unique, test_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"Cluster_ID {class_name}: " f"{count:,}")

model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist")

# =========================================================
# Print parameters
# =========================================================
print("\n====================================")
print("XGBoost Parameters")
print("====================================")
print(f"n_estimators: {model.n_estimators}")
print(f"max_depth: {model.max_depth}")
print(f"learning_rate: {model.learning_rate}")
print(f"subsample: {model.subsample}")
print(f"colsample_bytree: {model.colsample_bytree}")
print(f"tree_method: hist")
print(f"CPU threads: {model.n_jobs}")

# =========================================================
# Train
# =========================================================
print("\n====================================")
print("Training XGBoost")
print("====================================")
print("\nTraining progress will be printed " "every 10 boosting rounds.")
print("\nMonitoring train vs. validation loss (test set stays held out).")
print("\nStart training...\n")

start_time = time.time()
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],verbose=10)

elapsed_time = (time.time() - start_time)

print("\n====================================")
print("Training completed")
print("====================================")
print(f"Training time : " f"{elapsed_time:.2f} seconds")

# =========================================================
# Training history
# =========================================================
results = model.evals_result()
train_loss = results["validation_0"]["mlogloss"]
val_loss = results["validation_1"]["mlogloss"]
print("\nFinal training loss:")
print(f"{train_loss[-1]:.6f}")
print("\nFinal validation loss:")
print(f"{val_loss[-1]:.6f}")

# =========================================================
# Prediction
# =========================================================
print("\n====================================")
print("Prediction")
print("====================================")

prediction_start = time.time()
y_pred = model.predict(X_test)
prediction_time = (time.time() - prediction_start)
print(f"Predicted samples : " f"{len(y_pred):,}")
print(f"Prediction time   : " f"{prediction_time:.4f} seconds")

# =========================================================
# Accuracy
# =========================================================
accuracy = accuracy_score(y_test, y_pred)
print("\n====================================")
print("XGBoost Results")
print("====================================")
print(f"\nAccuracy: " f"{accuracy:.4f}")

# =========================================================
# Classification report
# =========================================================
target_names = [f"Cluster_{c}" for c in label_encoder.classes_]

print("\n====================================")
print("Classification Report")
print("====================================\n")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# =========================================================
# Confusion matrix
# =========================================================
cm = confusion_matrix(y_test, y_pred)
print("\n====================================")
print("Confusion Matrix")
print("====================================\n")
print(cm)

# =========================================================
# Confusion matrix with labels
# =========================================================
cm_df = pd.DataFrame(cm, index=[f"True_{name}" for name in target_names], columns=[f"Pred_{name}" for name in target_names])

print("\nConfusion Matrix with class names:\n")
print(cm_df)

# =========================================================
# Per-class TP / FN / FP / TN (one-vs-rest)
# =========================================================
num_classes = len(label_encoder.classes_)
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
# Feature importance
# =========================================================
print("\n====================================")
print("Feature Importance")
print("====================================")

feature_importance = pd.DataFrame({"Feature": feature_columns, "Importance": (model.feature_importances_)})

feature_importance = (feature_importance.sort_values("Importance", ascending=False))
print(feature_importance.to_string(index=False))

# =========================================================
# First 10 prediction examples
# =========================================================
predicted_classes = (label_encoder.inverse_transform(y_pred[:10]))
true_classes = (label_encoder.inverse_transform(y_test[:10]))
example_df = pd.DataFrame(X_test[:10], columns=feature_columns)
example_df["True"] = true_classes
example_df["Predicted"] = predicted_classes

print("\n====================================")
print("First 10 Prediction Examples")
print("====================================\n")
print(example_df.to_string(index=False))

# =========================================================
# Number of correct / incorrect predictions
# =========================================================
correct = np.sum(y_test == y_pred)
incorrect = np.sum(y_test != y_pred)
print("\n====================================")
print("Prediction Summary")
print("====================================")
print(f"Correct predictions   : " f"{correct:,}")
print(f"Incorrect predictions : " f"{incorrect:,}")
print(f"Total predictions     : " f"{len(y_test):,}")

# =========================================================
# Save trained model
# =========================================================
model.save_model(model_path)

print("\n====================================")
print("Model Saved")
print("====================================")
print(f"Saved model: " f"{model_path}")
print("\nDone.")
