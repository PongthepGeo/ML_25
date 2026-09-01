import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# =========================================================
# Paths
# =========================================================
image_path = Path("data/img_0037.png")
label_path = Path("data/label_0037.png")
output_extraction = Path("02_extraction.png")
output_distribution = Path("02_distribution.png")

# =========================================================
# Gaussian PDF
# =========================================================
def gaussian_pdf(x, mean, std):
    if std <= 0:
        return np.zeros_like(x)
    return (
        1.0 / (std * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std) ** 2)
    )

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
# Find closest class name
# =========================================================
def nearest_color_name(rgb):
    rgb = np.asarray(rgb, dtype=float)
    distances = {
        name: np.linalg.norm(rgb - ref)
        for name, ref in reference_colors.items()
    }

    return min(distances, key=distances.get)

# =========================================================
# Load image and label
# =========================================================
image = np.array(
    Image.open(image_path).convert("RGB")
)

label = np.array(
    Image.open(label_path).convert("RGB")
)


# =========================================================
# Check dimensions
# =========================================================
if image.shape[:2] != label.shape[:2]:

    raise ValueError(
        f"Image and label dimensions do not match.\n"
        f"Image: {image.shape}\n"
        f"Label: {label.shape}"
    )


print("Image shape :", image.shape)
print("Label shape :", label.shape)


# =========================================================
# Detect label colors
# =========================================================
unique_colors, counts = np.unique(
    label.reshape(-1, 3),
    axis=0,
    return_counts=True
)


print("\nDetected label colors:")

class_info = []


for rgb, count in zip(unique_colors, counts):

    class_name = nearest_color_name(rgb)

    print(
        f"{class_name:8s} "
        f"RGB={tuple(rgb)} "
        f"N={count:,}"
    )

    class_info.append(
        {
            "name": class_name,
            "rgb": rgb,
            "count": count,
        }
    )


# =========================================================
# Sort classes
# =========================================================
class_order = {
    "Grey": 0,
    "Blue": 1,
    "Green": 2,
    "Yellow": 3,
}


class_info.sort(
    key=lambda x: class_order.get(x["name"], 99)
)


# =========================================================
# Store extracted pixels
# =========================================================
class_pixels = {}

for info in class_info:

    name = info["name"]
    rgb = info["rgb"]

    mask = np.all(
        label == rgb,
        axis=2
    )

    class_pixels[name] = image[mask].astype(np.float64)


# =========================================================
# FIGURE 1
# AOI extraction
# =========================================================
fig, axes = plt.subplots(
    1,
    len(class_info),
    figsize=(16, 4)
)


for ax, info in zip(axes, class_info):

    name = info["name"]
    rgb = info["rgb"]

    mask = np.all(
        label == rgb,
        axis=2
    )

    # Black background
    extracted = np.zeros_like(image)

    # Original image only inside class AOI
    extracted[mask] = image[mask]

    ax.imshow(extracted)

    ax.set_title(
        f"{name}\n"
        f"N = {mask.sum():,}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    output_extraction,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# FIGURE 2
#
# Overlay ALL classes in the same distribution figure
#
# R | G | B
# =========================================================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)


band_names = [
    "Red",
    "Green",
    "Blue",
]


# Plot colors representing LABEL CLASSES
class_plot_colors = {
    "Grey": "gray",
    "Blue": "blue",
    "Green": "green",
    "Yellow": "gold",
}


x = np.linspace(
    0,
    255,
    1000
)


# =========================================================
# Loop through RGB bands
# =========================================================
for band_idx, ax in enumerate(axes):

    band_name = band_names[band_idx]


    # -----------------------------------------------------
    # Overlay all four classes
    # -----------------------------------------------------
    for info in class_info:

        class_name = info["name"]

        pixels = class_pixels[class_name]

        if len(pixels) == 0:
            continue


        values = pixels[:, band_idx]


        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------
        mean = np.mean(values)

        std = np.std(
            values,
            ddof=1
        )


        # -------------------------------------------------
        # Gaussian distribution
        # -------------------------------------------------
        y = gaussian_pdf(
            x,
            mean,
            std
        )


        ax.plot(
            x,
            y,
            linewidth=2.2,
            color=class_plot_colors[class_name],
            label=(
                f"{class_name}: "
                f"μ={mean:.1f}, "
                f"σ={std:.1f}"
            )
        )


        print(
            f"{class_name:8s} | "
            f"{band_name:5s} | "
            f"mean={mean:.2f}, "
            f"std={std:.2f}"
        )


    # -----------------------------------------------------
    # Format
    # -----------------------------------------------------
    ax.set_xlim(
        0,
        255
    )

    ax.set_xlabel(
        "Pixel value"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        f"{band_name} band"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        fontsize=9
    )


# =========================================================
# Save distribution figure
# =========================================================
plt.tight_layout()

plt.savefig(
    output_distribution,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(f"\nSaved: {output_extraction}")
print(f"Saved: {output_distribution}")