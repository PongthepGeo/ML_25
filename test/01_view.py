from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# Input
# ============================================================
IMAGE_PATH = Path("img.png")
LABEL_PATH = Path("label.png")


# ============================================================
# Load image and label
# ============================================================
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
label = np.array(Image.open(LABEL_PATH))

# print(image)

red = image[:, :, 0]
red = red.flatten()
print(red.shape)




# If label is RGB/RGBA, keep one channel if channels are identical
# if label.ndim == 3:
#     if label.shape[2] == 4:
#         label = label[:, :, :3]

#     if np.all(label[:, :, 0] == label[:, :, 1]) and \
#        np.all(label[:, :, 0] == label[:, :, 2]):
#         label = label[:, :, 0]


# # ============================================================
# # Check shapes
# # ============================================================
# print("=" * 60)
# print("IMAGE / LABEL SHAPE CHECK")
# print("=" * 60)

# print(f"Image path  : {IMAGE_PATH}")
# print(f"Label path  : {LABEL_PATH}")
# print(f"Image shape : {image.shape}")
# print(f"Label shape : {label.shape}")

# image_hw = image.shape[:2]
# label_hw = label.shape[:2]

# if image_hw == label_hw:
#     print("Spatial shape: MATCH")
# else:
#     print("Spatial shape: MISMATCH")
#     print(f"Image H x W : {image_hw}")
#     print(f"Label H x W : {label_hw}")

#     raise ValueError(
#         "Image and label spatial dimensions do not match."
#     )

# print("=" * 60)


# # ============================================================
# # Figure 1: Image and label side-by-side
# # ============================================================
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# axes[0].imshow(image)
# axes[0].set_title("Image")
# axes[0].axis("off")

# if label.ndim == 2:
#     axes[1].imshow(label, cmap="gray", interpolation="nearest")
# else:
#     axes[1].imshow(label, interpolation="nearest")

# axes[1].set_title("Label")
# axes[1].axis("off")

# plt.tight_layout()
# plt.savefig(
#     "figure_1_image_label.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()
# plt.close()


# # ============================================================
# # Figure 2: Label overlay on image
# # ============================================================
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.imshow(image)

# if label.ndim == 2:
#     # Do not overlay background pixels
#     overlay = np.ma.masked_where(label == 0, label)

#     ax.imshow(
#         overlay,
#         cmap="jet",
#         alpha=0.45,
#         interpolation="nearest"
#     )
# else:
#     ax.imshow(
#         label,
#         alpha=0.45,
#         interpolation="nearest"
#     )

# ax.set_title("Label Overlay")
# ax.axis("off")

# plt.tight_layout()
# plt.savefig(
#     "figure_2_label_overlay.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()
# plt.close()