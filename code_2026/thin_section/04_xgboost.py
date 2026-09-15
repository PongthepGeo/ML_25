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
# Train / validation data : img_0037, extracted to CSV by
#                           03_image2csv.py (default args).
# Test data                : img_0145, a different image,
#                           held out entirely. Extracted the
#                           same way, via:
#                             python 03_image2csv.py \
#                               --image data/img_0145.png \
#                               --label data/label_0145.png \
#                               --output-csv 03_image2csv/03_image2csv_test.csv
# =========================================================
csv_path = Path("03_image2csv/03_image2csv.csv")
test_csv_path = Path("03_image2csv/03_image2csv_test.csv")
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

class_counts = df["Class"].value_counts()
print(class_counts)
print("\nClass percentages:")

class_percentages = (df["Class"].value_counts(normalize=True).mul(100))

for class_name, percentage in class_percentages.items():
    print(f"{class_name:8s}: " f"{percentage:.2f}%")

# =========================================================
# Features
# =========================================================
feature_columns = ["R", "G", "B"]
X = df[feature_columns].values
y_text = df["Class"].values
print("\n====================================")
print("Features")
print("====================================")
print(feature_columns)
print(f"X shape: {X.shape}")

# =========================================================
# Encode classes
# =========================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
print("\n====================================")
print("Class encoding")
print("====================================")

for class_id, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name:8s} -> {class_id}")

# =========================================================
# Train / validation split
#
# img_0037 supplies ALL training + validation data here.
# eval_set below monitors train vs. validation loss during
# boosting -- both still come from img_0037.
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n====================================")
print("Train / Validation Split (img_0037)")
print("====================================")
print(f"Training samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")

print("\nTraining class counts:")

train_unique, train_counts = np.unique(y_train, return_counts=True)

for class_id, count in zip(train_unique, train_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"{class_name:8s}: " f"{count:,}")

print("\nValidation class counts:")
val_unique, val_counts = np.unique(y_val, return_counts=True)

for class_id, count in zip(val_unique, val_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"{class_name:8s}: " f"{count:,}")

# =========================================================
# Test set -- a DIFFERENT image (img_0145), held out entirely.
#
# This is never touched during training or validation above.
# Evaluating on a separate image (rather than a random split
# of img_0037's own pixels) is what actually measures
# generalization to new data. Extracted the same way as the
# training data, via 03_image2csv.py (see Paths above).
# =========================================================
print("\n====================================")
print("Load test dataset (img_0145)")
print("====================================")

df_test = pd.read_csv(test_csv_path)
print(f"CSV file: {test_csv_path}")
print(f"Dataset rows: {len(df_test):,}")

X_test = df_test[feature_columns].values
y_test = label_encoder.transform(df_test["Class"].values)

print(f"\nTesting samples : {len(X_test):,}")

print("\nTesting class counts:")
test_unique, test_counts = np.unique(y_test, return_counts=True)

for class_id, count in zip(test_unique, test_counts):
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"{class_name:8s}: " f"{count:,}")

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
print(f"Prediction time   : " f"{prediction_time:.2f} seconds")

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
print("\n====================================")
print("Classification Report")
print("====================================\n")
print(classification_report(y_test, y_pred, target_names=(label_encoder.classes_), digits=4))

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
cm_df = pd.DataFrame(cm, index=[f"True_{name}" for name in label_encoder.classes_], columns=[f"Pred_{name}" for name in label_encoder.classes_])

print("\nConfusion Matrix with class names:\n")
print(cm_df)

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