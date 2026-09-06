import rasterio
# pip install rasterio

import numpy as np
from PIL import Image
# pip install pillow

from pathlib import Path


# ============================================================
# OUTPUT
# ============================================================
OUTDIR = Path("05_satellite_composite")
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
# TIFF band order:
# Band 1 = B2 Blue
# Band 2 = B3 Green
# Band 3 = B4 Red
# Band 4 = B8 NIR
# ============================================================
with rasterio.open(tiff_file) as src:

    print(f"Image width  : {src.width}")
    print(f"Image height : {src.height}")
    print(f"Bands        : {src.count}")
    print(f"CRS          : {src.crs}")

    blue = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    red = src.read(3).astype(np.float32)


# ============================================================
# PERCENTILE STRETCH
# ============================================================
def percentile_stretch(
    channel,
    low=2,
    high=98
):

    valid = np.isfinite(channel)

    if not np.any(valid):
        return np.zeros(
            channel.shape,
            dtype=np.uint8
        )

    vmin, vmax = np.percentile(
        channel[valid],
        [low, high]
    )

    if vmax <= vmin:

        vmin = np.nanmin(
            channel[valid]
        )

        vmax = np.nanmax(
            channel[valid]
        )

    if vmax <= vmin:

        return np.zeros(
            channel.shape,
            dtype=np.uint8
        )

    stretched = (
        (channel - vmin)
        /
        (vmax - vmin)
    )

    stretched = np.clip(
        stretched,
        0,
        1
    )

    return (
        stretched * 255
    ).astype(np.uint8)


# ============================================================
# STRETCH RGB
# ============================================================
red_8 = percentile_stretch(red)

green_8 = percentile_stretch(green)

blue_8 = percentile_stretch(blue)


# ============================================================
# NATURAL COLOR
#
# Sentinel-2:
# R = B4
# G = B3
# B = B2
# ============================================================
rgb = np.dstack(
    (
        red_8,
        green_8,
        blue_8
    )
)


print(
    f"RGB shape     : {rgb.shape}"
)

print(
    f"RGB dtype     : {rgb.dtype}"
)

print(
    f"RGB range     : "
    f"{rgb.min()} - {rgb.max()}"
)


# ============================================================
# SAVE PNG
# ============================================================
output_png = (
    OUTDIR
    / "rgb_composite.png"
)


Image.fromarray(
    rgb,
    mode="RGB"
).save(
    output_png,
    format="PNG",
    optimize=True
)


print(
    f"Saved RGB composite: "
    f"{output_png}"
)