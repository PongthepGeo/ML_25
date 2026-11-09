import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import ndimage as ndi

# =========================================================
# Default paths (relative to this script, so it runs from any folder)
# =========================================================
# Two different photos, so the test set is a completely unseen image:
#   image 1 -> train_val.csv   (02_split.py then splits it into train / val)
#   image 2 -> test.csv        (never used for training or validation)
SOIL_DIR = Path(__file__).resolve().parent / "soil"
DEFAULT_TRAIN_VAL_IMAGE = SOIL_DIR / "34691_0_img.jpg"
DEFAULT_TRAIN_VAL_LABEL = SOIL_DIR / "34691_0_label_ดินทราย.jpg"
DEFAULT_TEST_IMAGE = SOIL_DIR / "34696_0_img.jpg"
DEFAULT_TEST_LABEL = SOIL_DIR / "34696_0_label_ดินทราย.jpg"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "01_image_to_table"
TRAIN_VAL_FILE = "train_val.csv"
TEST_FILE = "test.csv"

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
# <id>_label_<class name>.jpg  ->  <class name>
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
    print(f"Unique RGB colors in label: {len(unique_colors):,}")

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
    print(f"Soil objects in label: {n_objects} | areas (px): "
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
# One image/label pair -> one CSV table (with QC printout)
# =========================================================
def convert_and_save(title, image_path, label_path, class_name, output_csv):
    print(f"\n==================== {title} ====================")
    print(f"Image: {image_path}")
    print(f"Label: {label_path}")

    df, (height, width) = build_dataframe(image_path, label_path, class_name)

    print("\nClass counts:")
    print(df["Class"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head(5))
    print(f"\nData shape: {df.shape} | rows = height x width = {height} x {width} = {height * width:,}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")
    return df

# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert two image/label pairs into two (Row, Col, R, G, B, Class) tables: "
                    "train_val.csv (image 1, split into train/val by 02_split.py) and "
                    "test.csv (image 2, used only for testing)."
    )
    parser.add_argument("--train-val-image", type=str, default=str(DEFAULT_TRAIN_VAL_IMAGE))
    parser.add_argument("--train-val-label", type=str, default=str(DEFAULT_TRAIN_VAL_LABEL))
    parser.add_argument("--test-image", type=str, default=str(DEFAULT_TEST_IMAGE))
    parser.add_argument("--test-label", type=str, default=str(DEFAULT_TEST_LABEL))
    parser.add_argument("--class-name", type=str, default=None,
                        help="Name of the labeled class (default: taken from each label file name)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()

def main():
    args = parse_args()

    train_val_image, train_val_label = Path(args.train_val_image), Path(args.train_val_label)
    test_image, test_label = Path(args.test_image), Path(args.test_label)
    output_dir = Path(args.output_dir)

    # The whole point is an unseen test image: refuse to use the same photo twice
    if train_val_image.resolve() == test_image.resolve():
        raise ValueError(
            "The train/val image and the test image are the same file:\n"
            f"  {train_val_image}\nPick two different images."
        )

    df_train_val = convert_and_save(
        "IMAGE 1: train + val", train_val_image, train_val_label,
        args.class_name or class_name_from_label(train_val_label),
        output_dir / TRAIN_VAL_FILE,
    )
    df_test = convert_and_save(
        "IMAGE 2: test", test_image, test_label,
        args.class_name or class_name_from_label(test_label),
        output_dir / TEST_FILE,
    )

    # =========================================================
    # QC: size and class balance of the two tables
    # =========================================================
    print("\n==================== Summary ====================")
    for name, df in [(TRAIN_VAL_FILE, df_train_val), (TEST_FILE, df_test)]:
        share = 100 * df["Class"].value_counts(normalize=True)
        print(f"{name:14s} | rows: {len(df):>9,} | "
              + " | ".join(f"{cls}: {pct:.2f}%" for cls, pct in share.items()))

if __name__ == "__main__":
    main()
