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
DEFAULT_IMAGE = SCRIPT_DIR / "soil" / "34696_0_img.jpg"
DEFAULT_LABEL = SCRIPT_DIR / "soil" / "34696_0_label_ดินทราย.jpg"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "01_image_to_table"
TRAIN_VAL_FILE = "train_val.csv"
TEST_FILE = "test.csv"

# =========================================================
# Split by object (not by random pixel)
# =========================================================
#   the N_CIRCLES largest soil objects (the 2 circles) -> train_val.csv
#   every other soil object (the small blobs)          -> test.csv
# Each object keeps the background within MARGIN pixels around it.
N_CIRCLES = 2
MARGIN = 50

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
# 34696_0_label_<class name>.jpg  ->  <class name>
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
# Split the image by object: circles (train/val) vs small blobs (test)
# =========================================================
def split_regions(soil_mask, n_circles, margin):
    """
    Finds the connected soil objects, ranks them by area, and returns
    a region map (H, W):
        1 = train/val (the n_circles largest objects + margin)
        2 = test      (all other objects + margin)
        0 = unused    (farther than margin from every object)
    Each pixel goes to its nearest object, so the regions never overlap.
    """
    labels, n_objects = ndi.label(soil_mask, structure=np.ones((3, 3)))
    if n_objects <= n_circles:
        raise ValueError(f"Found {n_objects} soil objects; need more than {n_circles}.")

    areas = ndi.sum(soil_mask, labels, index=np.arange(1, n_objects + 1))
    ranked = np.argsort(-areas) + 1                      # component ids, largest first

    objects = np.zeros_like(labels)                      # 1 = largest, 2 = second, ...
    print("\nSoil objects (largest first):")
    for new_id, comp in enumerate(ranked, start=1):
        objects[labels == comp] = new_id
        ys, xs = np.where(labels == comp)
        role = "train_val (circle)" if new_id <= n_circles else "test (blob)"
        print(f"  object {new_id}: area={int(areas[comp - 1]):>7,} px | "
              f"x[{xs.min()}, {xs.max()}] y[{ys.min()}, {ys.max()}] -> {role}")

    # Nearest object (and distance to it) for every pixel
    distance, (iy, ix) = ndi.distance_transform_edt(objects == 0, return_indices=True)
    nearest = objects[iy, ix]
    near = distance <= margin

    region = np.zeros(soil_mask.shape, dtype=np.uint8)
    region[near & (nearest <= n_circles)] = 1
    region[near & (nearest > n_circles)] = 2
    return region

# =========================================================
# Build the (Row, Col, R, G, B, Class) tables: one row per pixel
# =========================================================
def build_dataframes(image_path, label_path, class_name, n_circles, margin):
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

    # -----------------------------------------------------
    # Split by object: circles -> train_val, small blobs -> test
    # -----------------------------------------------------
    region = split_regions((classes != "Background").reshape(height, width), n_circles, margin).reshape(-1)
    df_train_val = df[region == 1].reset_index(drop=True)
    df_test = df[region == 2].reset_index(drop=True)

    print(f"\nMargin: {margin} px")
    print(f"Unused (far background): {int((region == 0).sum()):,} px")

    return df_train_val, df_test, width

# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an image/label pair into two (Row, Col, R, G, B, Class) tables: "
                    "train_val.csv (circles) and test.csv (small blobs)."
    )
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--label", type=str, default=str(DEFAULT_LABEL))
    parser.add_argument("--class-name", type=str, default=None,
                        help="Name of the labeled class (default: taken from the label file name)")
    parser.add_argument("--n-circles", type=int, default=N_CIRCLES,
                        help="Number of largest soil objects used for train/val")
    parser.add_argument("--margin", type=int, default=MARGIN,
                        help="Background (pixels) kept around each object")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()

def main():
    args = parse_args()

    image_path = Path(args.image)
    label_path = Path(args.label)
    output_dir = Path(args.output_dir)
    class_name = args.class_name or class_name_from_label(label_path)

    df_train_val, df_test, width = build_dataframes(
        image_path, label_path, class_name, args.n_circles, args.margin
    )

    # =========================================================
    # QC
    # =========================================================
    for name, df in [(TRAIN_VAL_FILE, df_train_val), (TEST_FILE, df_test)]:
        print(f"\n===== {name} =====")
        print("Class counts:")
        print(df["Class"].value_counts())
        print("First 5 rows:")
        print(df.head(5))
        print("Data shape:", df.shape)

    # The two tables must never share a pixel
    key_train_val = df_train_val["Row"].to_numpy() * width + df_train_val["Col"].to_numpy()
    key_test = df_test["Row"].to_numpy() * width + df_test["Col"].to_numpy()
    print(f"\nPixels shared by {TRAIN_VAL_FILE} and {TEST_FILE}: "
          f"{np.intersect1d(key_train_val, key_test).size}")

    # =========================================================
    # Save CSV
    # =========================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train_val.to_csv(output_dir / TRAIN_VAL_FILE, index=False)
    df_test.to_csv(output_dir / TEST_FILE, index=False)
    print(f"\nSaved CSV: {output_dir / TRAIN_VAL_FILE}")
    print(f"Saved CSV: {output_dir / TEST_FILE}")

if __name__ == "__main__":
    main()
