import rasterio
# pip install rasterio
import skimage.io as skio
# pip install scikit-image
# pip install imagecodecs
import matplotlib.pyplot as plt
import os

tiff_file = 'dataset/sen2_2024_04_12.tif'

header_information = rasterio.open(tiff_file).profile
# print(header_information)
img = skio.imread(tiff_file, plugin='tifffile')
# print(f'Image shape: {img.shape}')

# The orginal TIFF contains B2 (blue), B3 (green), B4 (red), and B8 bands.
red = img[:, :, 2]; green = img[:, :, 1]; blue = img[:, :, 0]
# Create a list of channels with their titles and colormaps
channels = [
    (red, 'Red Channel', 'Reds'),
    (green, 'Green Channel', 'Greens'),
    (blue, 'Blue Channel', 'Blues')
]

# Plot the separate red, green, and blue channels
fig, ax = plt.subplots(1, 3, figsize=(25, 15))
for i, (channel, title, cmap) in enumerate(channels):
    ax[i].imshow(channel, cmap=cmap)
    ax[i].set_title(title)
    ax[i].axis('off')
os.makedirs('figure_plot', exist_ok=True)
plt.savefig('figure_plot/' + 'satellite' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()