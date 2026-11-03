import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import ndimage as ndi

# =========================================================
# Default paths (relative to this script, so it runs from any folder)
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "34694_0_img.jpg"
DEFAULT_LABEL = SCRIPT_DIR / "34694_0_label_ดินทราย.jpg"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "01_image_to_table" / "01_image_to_table.csv"

# =========================================================
# Reference label colors
# =========================================================
# The label is a JPEG, so compression blurs the two drawn colors
# into thousands of near-white / near-magenta shades.
# Every label pixel is therefore assigned to the NEAREST color below.
#   White   = Background
#   Magenta = Soil (class name is read from the label file name)
BACKGROUND_RGB = np.array([255, 255, 255])
SOIL_RGB = np.array([255, 0, 255])

# =========================================================
# Class name from the label file name
# =========================================================
# 34694_0_label_<class name>.jpg  ->  <class name>
def class_name_from_label(label_path):
    stem = Path(label_path).stem
    if "_label_" in stem:
        return stem.split("_label_", 1)[1]
    return "Soil"

# =========================================================
# Assign each label pixel to its nearest reference color
# =========================================================
def classify_label(label, reference_colors):
    names = np.array(list(reference_colors.keys()))
    refs = np.stack(list(reference_colors.values())).astype(np.int32)
    pixels = label.reshape(-1, 3).astype(np.int32)

    # Squared RGB distance to every reference color -> (pixels, classes)
    distances = ((pixels[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    nearest_distance = np.sqrt(distances.min(axis=1))

    return names[nearest], nearest_distance

# =========================================================
# Build a (Row, Col, R, G, B, Class) table: one row per pixel
# =========================================================
def build_dataframe(image_path, label_path, class_name):
    image = np.array(Image.open(image_path).convert("RGB"))
    label = np.array(Image.open(label_path).convert("RGB"))

    if image.shape[:2] != label.shape[:2]:
        raise ValueError(
            "Image and label dimensions do not match.\n"
            f"Image: {image.shape}\n"
            f"Label: {label.shape}"
        )
    height, width = image.shape[:2]
    print("Image shape :", image.shape)
    print("Label shape :", label.shape)

    # -----------------------------------------------------
    # Detect label RGB colors (JPEG artifacts inflate this)
    # -----------------------------------------------------
    unique_colors = np.unique(label.reshape(-1, 3), axis=0)
    print(f"\nUnique RGB colors in label: {len(unique_colors):,}")

    # -----------------------------------------------------
    # Classify every pixel by nearest reference color
    # -----------------------------------------------------
    reference_colors = {
        "Background": BACKGROUND_RGB,
        class_name: SOIL_RGB,
    }
    classes, nearest_distance = classify_label(label, reference_colors)
    print(
        "Distance to nearest label color: "
        f"mean={nearest_distance.mean():.2f}, "
        f"max={nearest_distance.max():.1f}"
    )

    # -----------------------------------------------------
    # QC: connected soil objects in the label (dishes / lumps)
    # -----------------------------------------------------
    soil_mask = (classes != "Background").reshape(height, width)
    labels, n_objects = ndi.label(soil_mask, structure=np.ones((3, 3)))
    areas = sorted(ndi.sum(soil_mask, labels, index=np.arange(1, n_objects + 1)), reverse=True)
    print(f"\nSoil objects in label: {n_objects} | areas (px): "
          + ", ".join(f"{int(a):,}" for a in areas))

    # -----------------------------------------------------
    # Image pixels -> table rows (row-major order, top-left first)
    # Row / Col keep the pixel position, so predictions can be
    # placed back on the image after the table is split
    # -----------------------------------------------------
    rows, cols = np.divmod(np.arange(height * width), width)
    pixels = image.reshape(-1, 3)
    df = pd.DataFrame({
        "Row": rows,
        "Col": cols,
        "R": pixels[:, 0],
        "G": pixels[:, 1],
        "B": pixels[:, 2],
        "Class": classes,
    })

    return df, (height, width)

# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an image/label pair into one (Row, Col, R, G, B, Class) table "
                    "and save it as CSV. Split it into train/val/test with 02_split.py."
    )
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--label", type=str, default=str(DEFAULT_LABEL))
    parser.add_argument("--class-name", type=str, default=None,
                        help="Name of the labeled class (default: taken from the label file name)")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()

def main():
    args = parse_args()

    image_path = Path(args.image)
    label_path = Path(args.label)
    output_csv = Path(args.output_csv)
    class_name = args.class_name or class_name_from_label(label_path)

    df, (height, width) = build_dataframe(image_path, label_path, class_name)

    # =========================================================
    # QC
    # =========================================================
    print("\nClass counts:")
    print(df["Class"].value_counts())
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nData shape:")
    print(df.shape)
    print(f"Rows = height x width = {height} x {width} = {height * width:,}")

    # =========================================================
    # Save CSV
    # =========================================================
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV: {output_csv}")

if __name__ == "__main__":
    main()
