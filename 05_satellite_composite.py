import rasterio
# pip install rasterio

import skimage.io as skio
# pip install scikit-image
# pip install imagecodecs

import numpy as np
from PIL import Image
# pip install pillow

import os

tiff_file = 'dataset/sen2_2024_04_12.tif'

header_information = rasterio.open(tiff_file).profile
# print(header_information)

img = skio.imread(tiff_file, plugin='tifffile')
# print(f'Image shape: {img.shape}')

# Sentinel-2 band order in this TIFF:
# B2 = Blue  -> img[:, :, 0]
# B3 = Green -> img[:, :, 1]
# B4 = Red   -> img[:, :, 2]
# B8 = NIR   -> img[:, :, 3]

red   = img[:, :, 2].astype(np.float32)
green = img[:, :, 1].astype(np.float32)
blue  = img[:, :, 0].astype(np.float32)


def percentile_stretch(channel, low=2, high=98):
    """Stretch one band to 0-255 using percentile contrast enhancement."""
    valid = np.isfinite(channel)

    if not np.any(valid):
        return np.zeros(channel.shape, dtype=np.uint8)

    vmin, vmax = np.percentile(channel[valid], [low, high])

    if vmax <= vmin:
        vmin = np.nanmin(channel[valid])
        vmax = np.nanmax(channel[valid])

    if vmax <= vmin:
        return np.zeros(channel.shape, dtype=np.uint8)

    stretched = (channel - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0, 1)

    return (stretched * 255).astype(np.uint8)


# Stretch each visible band independently.
red_8   = percentile_stretch(red)
green_8 = percentile_stretch(green)
blue_8  = percentile_stretch(blue)

# Composite B4-B3-B2 as natural-color RGB.
rgb = np.dstack((red_8, green_8, blue_8))

# Save directly as PNG without matplotlib.
os.makedirs('figure_plot', exist_ok=True)

output_png = 'figure_plot/05_satellite.png'

Image.fromarray(rgb, mode='RGB').save(
    output_png,
    format='PNG',
    optimize=True
)

print(f'Saved RGB composite: {output_png}')
