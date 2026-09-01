from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# INPUT
# ============================================================
IMAGE_PATH = Path("img.png")
LABEL_PATH = Path("label.png")


# ============================================================
# LOAD IMAGE
# ============================================================
image = np.array(
    Image.open(IMAGE_PATH).convert("RGB")
)


# ============================================================
# LOAD LABEL WITHOUT .convert("L")
# Preserve raw pixel values
# ============================================================
label_raw = np.array(
    Image.open(LABEL_PATH)
)


# ============================================================
# RAW LABEL QC
# ============================================================
print("=" * 70)
print("RAW LABEL QC")
print("=" * 70)

print(f"Image shape     : {image.shape}")
print(f"Image dtype     : {image.dtype}")

print(f"Raw label shape : {label_raw.shape}")
print(f"Raw label dtype : {label_raw.dtype}")


# ============================================================
# HANDLE LABEL FORMAT
# ============================================================
if label_raw.ndim == 2:

    # Already single-channel
    label = label_raw.copy()

    print(
        "Unique raw label values:",
        np.unique(label)
    )


elif label_raw.ndim == 3:

    # --------------------------------------------------------
    # Show unique raw colors
    # --------------------------------------------------------
    unique_colors = np.unique(
        label_raw.reshape(
            -1,
            label_raw.shape[2]
        ),
        axis=0
    )

    print("Unique raw label colors:")
    print(unique_colors)


    # --------------------------------------------------------
    # Use first channel directly
    #
    # This preserves values such as 0 and 255
    # instead of converting RGB color to grayscale luminance.
    # --------------------------------------------------------
    label = label_raw[:, :, 0].copy()

    print(
        "Unique first-channel values:",
        np.unique(label)
    )


else:

    raise ValueError(
        f"Unsupported label shape: {label_raw.shape}"
    )


# ============================================================
# CHECK SPATIAL SHAPE
# ============================================================
if image.shape[:2] != label.shape[:2]:

    raise ValueError(
        f"Spatial shape mismatch:\n"
        f"Image = {image.shape[:2]}\n"
        f"Label = {label.shape[:2]}"
    )


print("Spatial shape   : MATCH")


# ============================================================
# LABEL DEFINITION
#
# 0   = AOI / TARGET
# 255 = BACKGROUND
# ============================================================
aoi_mask = label == 0


n_total = aoi_mask.size
n_aoi = np.sum(aoi_mask)


if n_aoi == 0:

    raise ValueError(
        "No AOI pixels found where label == 0."
    )


print(f"Total pixels    : {n_total}")
print(f"AOI pixels      : {n_aoi}")
print(f"AOI fraction    : {n_aoi / n_total:.6f}")

print("=" * 70)


# ============================================================
# FIGURE 1
# IMAGE + RAW LABEL
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)


# RGB image
axes[0].imshow(image)
axes[0].set_title("RGB Image")
axes[0].axis("off")


# Label
axes[1].imshow(
    label,
    cmap="gray",
    vmin=0,
    vmax=255,
    interpolation="nearest"
)

axes[1].set_title("Raw Label")
axes[1].axis("off")


plt.tight_layout()

plt.savefig(
    "figure_1_image_label.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# FIGURE 2
# AOI OVERLAY ON RGB IMAGE
# ============================================================
fig, ax = plt.subplots(
    figsize=(8, 8)
)


ax.imshow(image)


# Mask background.
# Only label == 0 remains visible.
overlay = np.ma.masked_where(
    ~aoi_mask,
    aoi_mask.astype(float)
)


ax.imshow(
    overlay,
    cmap="Reds",
    alpha=0.45,
    interpolation="nearest",
    vmin=0,
    vmax=1
)


ax.set_title("Label AOI Overlay")
ax.axis("off")


plt.tight_layout()

plt.savefig(
    "figure_2_label_overlay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# EXTRACT RGB BANDS
# ============================================================
R = image[:, :, 0]
G = image[:, :, 1]
B = image[:, :, 2]


# Whole-image values
red = R.ravel()
green = G.ravel()
blue = B.ravel()


# ============================================================
# EXTRACT RGB VALUES INSIDE LABEL AOI
# ============================================================
red_aoi = R[aoi_mask].ravel()
green_aoi = G[aoi_mask].ravel()
blue_aoi = B[aoi_mask].ravel()


# ============================================================
# NORMAL DISTRIBUTION FUNCTION
# ============================================================
def normal_pdf(x, mean, std):

    if std <= 0:
        return np.zeros_like(x)

    return (
        1.0
        / (std * np.sqrt(2.0 * np.pi))
        * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )
    )


# ============================================================
# AOI RGB STATISTICS
# ============================================================
r_min = np.min(red_aoi)
r_max = np.max(red_aoi)
r_mean = np.mean(red_aoi)
r_std = np.std(red_aoi)

g_min = np.min(green_aoi)
g_max = np.max(green_aoi)
g_mean = np.mean(green_aoi)
g_std = np.std(green_aoi)

b_min = np.min(blue_aoi)
b_max = np.max(blue_aoi)
b_mean = np.mean(blue_aoi)
b_std = np.std(blue_aoi)


print()
print("=" * 70)
print("AOI RGB STATISTICS")
print("=" * 70)

print(
    f"RED   : "
    f"min={r_min:3d}, "
    f"max={r_max:3d}, "
    f"mean={r_mean:.2f}, "
    f"std={r_std:.2f}"
)

print(
    f"GREEN : "
    f"min={g_min:3d}, "
    f"max={g_max:3d}, "
    f"mean={g_mean:.2f}, "
    f"std={g_std:.2f}"
)

print(
    f"BLUE  : "
    f"min={b_min:3d}, "
    f"max={b_max:3d}, "
    f"mean={b_mean:.2f}, "
    f"std={b_std:.2f}"
)

print("=" * 70)


# ============================================================
# BAND CONFIGURATION
# ============================================================
bands = [

    (
        "Red",
        red,
        red_aoi,
        r_min,
        r_max,
        r_mean,
        r_std,
        "red"
    ),

    (
        "Green",
        green,
        green_aoi,
        g_min,
        g_max,
        g_mean,
        g_std,
        "green"
    ),

    (
        "Blue",
        blue,
        blue_aoi,
        b_min,
        b_max,
        b_mean,
        b_std,
        "blue"
    ),
]


# ============================================================
# FIGURE 3
#
# Whole-image histogram
# AOI histogram
# AOI normal distribution
# AOI RGB threshold range
# ============================================================
fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    sharex=True
)


x = np.linspace(
    0,
    255,
    1000
)


for ax, band in zip(
    axes,
    bands
):

    (
        name,
        all_values,
        aoi_values,
        threshold_min,
        threshold_max,
        aoi_mean,
        aoi_std,
        color
    ) = band


    # --------------------------------------------------------
    # Whole-image histogram
    # --------------------------------------------------------
    ax.hist(
        all_values,
        bins=64,
        range=(0, 255),
        density=True,
        color=color,
        alpha=0.20,
        edgecolor="none",
        label="Whole image"
    )


    # --------------------------------------------------------
    # AOI histogram
    # --------------------------------------------------------
    ax.hist(
        aoi_values,
        bins=64,
        range=(0, 255),
        density=True,
        histtype="step",
        linewidth=1.8,
        color="black",
        label="Label AOI"
    )


    # --------------------------------------------------------
    # AOI fitted normal distribution
    # --------------------------------------------------------
    pdf_aoi = normal_pdf(
        x,
        aoi_mean,
        aoi_std
    )


    ax.plot(
        x,
        pdf_aoi,
        color=color,
        linewidth=2.5,
        label=(
            rf"AOI normal "
            rf"$\mu={aoi_mean:.1f}$, "
            rf"$\sigma={aoi_std:.1f}$"
        )
    )


    # --------------------------------------------------------
    # AOI RGB RANGE
    #
    # This is the RGB threshold derived directly
    # from pixels inside label == 0.
    # --------------------------------------------------------
    ax.axvspan(
        threshold_min,
        threshold_max,
        color=color,
        alpha=0.18,
        label=(
            f"AOI range "
            f"[{threshold_min}, {threshold_max}]"
        )
    )


    # Lower RGB threshold
    ax.axvline(
        threshold_min,
        color="black",
        linestyle="--",
        linewidth=1.2
    )


    # Upper RGB threshold
    ax.axvline(
        threshold_max,
        color="black",
        linestyle="--",
        linewidth=1.2
    )


    # AOI mean
    ax.axvline(
        aoi_mean,
        color=color,
        linestyle=":",
        linewidth=2
    )


    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        f"{name} Band"
    )

    ax.legend(
        loc="upper right",
        fontsize=8
    )

    ax.grid(
        alpha=0.2
    )


axes[-1].set_xlabel(
    "Pixel intensity"
)

axes[-1].set_xlim(
    0,
    255
)


plt.tight_layout()

plt.savefig(
    "figure_3_rgb_aoi_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# RGB THRESHOLD FROM LABEL AOI
#
# Pixel is selected if it satisfies ALL THREE ranges:
#
# Rmin <= R <= Rmax
# Gmin <= G <= Gmax
# Bmin <= B <= Bmax
# ============================================================
rgb_threshold_mask = (

    (R >= r_min)
    & (R <= r_max)

    & (G >= g_min)
    & (G <= g_max)

    & (B >= b_min)
    & (B <= b_max)

)


# ============================================================
# FIGURE 4
# COMPARE LABEL AOI AND RGB THRESHOLD
# ============================================================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 6)
)


# ------------------------------------------------------------
# RGB image
# ------------------------------------------------------------
axes[0].imshow(image)

axes[0].set_title(
    "RGB Image"
)

axes[0].axis("off")


# ------------------------------------------------------------
# Ground-truth AOI
# ------------------------------------------------------------
axes[1].imshow(image)


gt_overlay = np.ma.masked_where(
    ~aoi_mask,
    aoi_mask.astype(float)
)


axes[1].imshow(
    gt_overlay,
    cmap="Reds",
    alpha=0.5,
    interpolation="nearest",
    vmin=0,
    vmax=1
)


axes[1].set_title(
    "Label AOI"
)

axes[1].axis("off")


# ------------------------------------------------------------
# RGB threshold result
# ------------------------------------------------------------
axes[2].imshow(image)


threshold_overlay = np.ma.masked_where(
    ~rgb_threshold_mask,
    rgb_threshold_mask.astype(float)
)


axes[2].imshow(
    threshold_overlay,
    cmap="Reds",
    alpha=0.5,
    interpolation="nearest",
    vmin=0,
    vmax=1
)


axes[2].set_title(
    "RGB Threshold from AOI"
)

axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    "figure_4_rgb_threshold_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# FIGURE 5
# BINARY RGB THRESHOLD MASK
# ============================================================
fig, ax = plt.subplots(
    figsize=(8, 8)
)


ax.imshow(
    rgb_threshold_mask,
    cmap="gray",
    interpolation="nearest"
)

ax.set_title(
    "RGB Threshold Mask"
)

ax.axis("off")


plt.tight_layout()

plt.savefig(
    "figure_5_rgb_threshold_mask.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 70)
print("FINAL RGB THRESHOLDS FROM LABEL AOI")
print("=" * 70)

print(
    f"{r_min} <= R <= {r_max}"
)

print(
    f"{g_min} <= G <= {g_max}"
)

print(
    f"{b_min} <= B <= {b_max}"
)

print()

print(
    "Label AOI pixels        :",
    np.sum(aoi_mask)
)

print(
    "RGB threshold pixels    :",
    np.sum(rgb_threshold_mask)
)

print(
    "RGB threshold fraction  :",
    np.mean(rgb_threshold_mask)
)

print("=" * 70)