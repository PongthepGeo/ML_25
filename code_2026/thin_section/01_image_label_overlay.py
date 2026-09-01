import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# =========================================================
# Paths
# =========================================================
image_path = Path("data/img_0037.png")
label_path = Path("data/label_0037.png")
output_path = Path("01_image_label_overlay.png")

# =========================================================
# Load image and RGB label
# =========================================================
image = np.array(Image.open(image_path).convert("RGB"))
label = np.array(Image.open(label_path).convert("RGB"))

# =========================================================
# Check dimensions
# =========================================================
if image.shape[:2] != label.shape[:2]:
    raise ValueError(
        f"Image and label dimensions do not match.\n"
        f"Image: {image.shape}\n"
        f"Label: {label.shape}"
    )

# =========================================================
# QC: print RGB colors present in label
# =========================================================
unique_colors = np.unique(label.reshape(-1, 3), axis=0)

print(f"Image shape : {image.shape}")
print(f"Label shape : {label.shape}")

print("\nUnique RGB label colors:")
for color in unique_colors:
    print(color)

# =========================================================
# Create figure
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---------------------------------------------------------
# Panel 1: Original image
# ---------------------------------------------------------
axes[0].imshow(image)
axes[0].set_title("Image")
axes[0].axis("off")

# ---------------------------------------------------------
# Panel 2: Original RGB label
# grey / blue / green / yellow are preserved
# ---------------------------------------------------------
axes[1].imshow(label)
axes[1].set_title("Label")
axes[1].axis("off")

# ---------------------------------------------------------
# Panel 3: Image + RGB label overlay
# ---------------------------------------------------------
axes[2].imshow(image)
axes[2].imshow(
    label,
    alpha=0.45
)
axes[2].set_title("Image + Label Overlay")
axes[2].axis("off")

# =========================================================
# Layout and save
# =========================================================
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)
plt.show()