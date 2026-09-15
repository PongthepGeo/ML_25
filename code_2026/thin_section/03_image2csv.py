import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# =========================================================
# Reference label colors
# =========================================================
# Mineral identity behind each label color (ore microscopy):
#   Grey   = Background
#   Blue   = Pyrite
#   Green  = Sphalerite
#   Yellow = Galena
# Not present in the current labeled images (img_0037 / img_0145):
#   Red    = Chalcopyrite
#   Pink   = Gold
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
# Build a (R, G, B, Class) dataframe from one image/label pair
# =========================================================
def build_dataframe(image_path, label_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    label = np.array(Image.open(label_path).convert("RGB"))

    if image.shape[:2] != label.shape[:2]:
        raise ValueError(
            "Image and label dimensions do not match.\n"
            f"Image: {image.shape}\n"
            f"Label: {label.shape}"
        )
    print("Image shape :", image.shape)
    print("Label shape :", label.shape)

    # -----------------------------------------------------
    # Detect label RGB colors
    # -----------------------------------------------------
    unique_colors, counts = np.unique(
        label.reshape(-1, 3),
        axis=0,
        return_counts=True)
    print("\nDetected classes:")

    # -----------------------------------------------------
    # Extract RGB pixels for each class
    # -----------------------------------------------------
    dataframes = []

    for label_rgb, count in zip(
        unique_colors,
        counts
    ):
        # Determine class
        class_name = nearest_color_name(label_rgb)
        # AOI mask
        mask = np.all(label == label_rgb, axis=2)
        # Extract original image RGB values
        pixels = image[mask]
        print(
            f"{class_name:8s} | "
            f"Label RGB={tuple(label_rgb)} | "
            f"Pixels={len(pixels):,}")
        # Convert to dataframe
        df_class = pd.DataFrame(pixels, columns=["R", "G", "B"])
        # Add class label
        df_class["Class"] = class_name
        dataframes.append(df_class)

    # -----------------------------------------------------
    # Combine all classes
    # -----------------------------------------------------
    df = pd.concat(dataframes, ignore_index=True)

    # -----------------------------------------------------
    # Arrange columns
    # -----------------------------------------------------
    df = df[["R", "G", "B", "Class"]]

    return df

# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract (R, G, B, Class) pixels from an image/label pair into a CSV."
    )
    parser.add_argument("--image", type=str, default="data/img_0037.png")
    parser.add_argument("--label", type=str, default="data/label_0037.png")
    parser.add_argument("--output-csv", type=str, default="03_image2csv/03_image2csv.csv")
    return parser.parse_args()

def main():
    args = parse_args()

    image_path = Path(args.image)
    label_path = Path(args.label)
    output_csv = Path(args.output_csv)
    output_dir = output_csv.parent

    df = build_dataframe(image_path, label_path)

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
    # Save CSV
    # =========================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV: {output_csv}")

if __name__ == "__main__":
    main()
