import rasterio
# pip install rasterio

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from lib.control_plot import PLOT_PARAMS


# ============================================================
# GLOBAL PLOT STYLE
# ============================================================
matplotlib.rcParams.update(PLOT_PARAMS)


# ============================================================
# OUTPUT
# ============================================================
OUTDIR = Path("05_satellite")
OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT
# ============================================================
tiff_file = "dataset/sen2_2024_04_12.tif"


# ============================================================
# READ SENTINEL-2 TIFF
#
# Band 1 = B2 Blue
# Band 2 = B3 Green
# Band 3 = B4 Red
# Band 4 = B8 NIR
# ============================================================
with rasterio.open(tiff_file) as src:

    print("=" * 60)
    print("SATELLITE TIFF INFORMATION")
    print("=" * 60)

    print(f"Width  : {src.width}")
    print(f"Height : {src.height}")
    print(f"Bands  : {src.count}")
    print(f"CRS    : {src.crs}")
    print(f"Dtype  : {src.dtypes}")

    blue = src.read(1)
    green = src.read(2)
    red = src.read(3)


# ============================================================
# CHANNEL LIST
# ============================================================
channels = [
    (
        red,
        "Red Channel",
        "Reds"
    ),
    (
        green,
        "Green Channel",
        "Greens"
    ),
    (
        blue,
        "Blue Channel",
        "Blues"
    )
]


# ============================================================
# PLOT RGB CHANNELS
# ============================================================
fig, ax = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


for i, (
    channel,
    title,
    cmap
) in enumerate(channels):

    ax[i].imshow(
        channel,
        cmap=cmap
    )

    ax[i].set_title(
        title
    )

    ax[i].axis(
        "off"
    )


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================
fig_channels = (
    OUTDIR
    / "satellite_rgb_channels.png"
)


plt.savefig(
    fig_channels,
    format="png",
    bbox_inches="tight"
)


print(
    f"Saved figure: {fig_channels}"
)


plt.show()
plt.close()