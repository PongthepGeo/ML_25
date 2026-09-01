import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


# =========================================================
# Paths
# =========================================================
image_path = Path("data/img_0037.png")
label_path = Path("data/label_0037.png")
output_dir = Path("03_image2csv")
output_csv = output_dir / "03_image2csv.csv"

# =========================================================
# Reference label colors
# =========================================================
reference_colors = {
    "Grey": np.array([128, 128, 128]),
    "Blue": np.array([0, 0, 255]),
    "Green": np.array([0, 255, 0]),
    "Yellow": np.array([255, 255, 0]),
}

# =========================================================
# Find nearest label class
# =========================================================
def nearest_color_name(rgb):
    rgb = np.asarray(rgb, dtype=float)
    distances = {
        name: np.linalg.norm(
            rgb - ref.astype(float)
        )
        for name, ref in reference_colors.items()
    }
    return min(
        distances,
        key=distances.get
    )

# =========================================================
# Load image and label
# =========================================================
image = np.array(Image.open(image_path).convert("RGB"))
label = np.array(Image.open(label_path).convert("RGB"))

# =========================================================
# Check dimensions
# =========================================================
if image.shape[:2] != label.shape[:2]:
    raise ValueError(
        "Image and label dimensions do not match.\n"
        f"Image: {image.shape}\n"
        f"Label: {label.shape}"
    )
print("Image shape :", image.shape)
print("Label shape :", label.shape)

# =========================================================
# Detect label RGB colors
# =========================================================
unique_colors, counts = np.unique(
    label.reshape(-1, 3),
    axis=0,
    return_counts=True)
print("\nDetected classes:")

# =========================================================
# Extract RGB pixels for each class
# =========================================================
dataframes = []

for label_rgb, count in zip(
    unique_colors,
    counts
):
    # -----------------------------------------------------
    # Determine class
    # -----------------------------------------------------
    class_name = nearest_color_name(label_rgb)
    # -----------------------------------------------------
    # AOI mask
    # -----------------------------------------------------
    mask = np.all(label == label_rgb, axis=2)
    # -----------------------------------------------------
    # Extract original image RGB values
    # -----------------------------------------------------
    pixels = image[mask]
    print(
        f"{class_name:8s} | "
        f"Label RGB={tuple(label_rgb)} | "
        f"Pixels={len(pixels):,}")
    # -----------------------------------------------------
    # Convert to dataframe
    # -----------------------------------------------------
    df_class = pd.DataFrame(pixels, columns=["R", "G", "B"])
    # -----------------------------------------------------
    # Add class label
    # -----------------------------------------------------
    df_class["Class"] = class_name
    dataframes.append(df_class)

# =========================================================
# Combine all four classes
# =========================================================
df = pd.concat(dataframes, ignore_index=True)

# =========================================================
# Arrange columns
# =========================================================
df = df[["R", "G", "B", "Class"]]

# =========================================================
# QC
# =========================================================
print("\nClass counts:")
print(df["Class"].value_counts())
print("\nFirst rows:")
print(df.head())
print("\nData shape:")
print(df.shape)

# =========================================================
# Create output directory
# =========================================================
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Save CSV
# =========================================================
df.to_csv(output_csv, index=False)
print(f"\nSaved CSV: {output_csv}")

# =========================================================
# QC
# =========================================================
print("\nClass counts:")
print(df["Class"].value_counts())
print("\nFirst 10 rows:")
print(df.head(10))
print("\nData shape:")
print(df.shape)

# =========================================================
# Create output directory
# =========================================================
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Save CSV
# =========================================================
df.to_csv(output_csv, index=False)
print(f"\nSaved CSV: {output_csv}")