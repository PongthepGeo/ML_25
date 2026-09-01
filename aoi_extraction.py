from pathlib import Path
import numpy as np
from PIL import Image

IMG_PATH = Path(
    "/home/laptop_pt/Desktop/work/class/ML_25/main_code/"
    "figure_plot/05_satellite.png"
)

LABEL_PATH = Path(
    "/home/laptop_pt/Desktop/work/class/ML_25/main_code/"
    "figure_plot/label.png"
)

OUTPUT_PATH = Path(
    "/home/laptop_pt/Desktop/work/class/ML_25/main_code/"
    "figure_plot/aoi.png"
)


def main():
    img = Image.open(IMG_PATH).convert("RGB")
    label = Image.open(LABEL_PATH).convert("RGB")

    if img.size != label.size:
        raise ValueError(
            f"Image and label sizes differ:\n"
            f"  satellite: {img.size}\n"
            f"  label    : {label.size}"
        )

    img_np = np.asarray(img)
    label_np = np.asarray(label)

    # -------------------------------------------------------------
    # Single-class label:
    # assume the most frequent RGB value is background.
    # Every other color is treated as the AOI.
    # -------------------------------------------------------------
    flat = label_np.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    if len(colors) < 2:
        raise ValueError(
            "label.png contains only one RGB value over the entire image. "
            "A background/AOI boundary cannot be determined."
        )

    background_color = colors[np.argmax(counts)]
    mask = np.any(label_np != background_color, axis=2)

    if not np.any(mask):
        raise ValueError("No AOI pixels were found in label.png.")

    # Find AOI bounding box.
    y, x = np.where(mask)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Crop both satellite image and AOI mask.
    img_crop = img_np[y_min:y_max + 1, x_min:x_max + 1]
    mask_crop = mask[y_min:y_max + 1, x_min:x_max + 1]

    # Keep satellite RGB inside AOI and make non-AOI pixels transparent.
    alpha = np.where(mask_crop, 255, 0).astype(np.uint8)
    aoi_rgba = np.dstack((img_crop, alpha))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(aoi_rgba, mode="RGBA").save(
        OUTPUT_PATH,
        format="PNG",
        optimize=True
    )

    print(f"Satellite      : {IMG_PATH}")
    print(f"Label          : {LABEL_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Original size  : {img.width} x {img.height}")
    print(f"AOI size       : {aoi_rgba.shape[1]} x {aoi_rgba.shape[0]}")
    print(f"Background RGB : {tuple(background_color.tolist())}")
    print(f"AOI pixels     : {mask.sum():,}")
    print(
        f"AOI bbox       : "
        f"x={x_min}:{x_max}, y={y_min}:{y_max}"
    )


if __name__ == "__main__":
    main()
