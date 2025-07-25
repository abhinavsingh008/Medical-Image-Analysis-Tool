import os
from PIL import Image # type: ignore
import numpy as np # type: ignore

# Paths
left_mask_dir = "C:/Users/Abhinav Singh/Desktop/lung_segmentation_project/data/MontgomerySet/ManualMask/leftMask"
right_mask_dir = "C:/Users/Abhinav Singh/Desktop/lung_segmentation_project/data/MontgomerySet/ManualMask/rightMask"
output_mask_dir = "C:/Users/Abhinav Singh/Desktop/lung_segmentation_project/data/masks"
os.makedirs(output_mask_dir, exist_ok=True)

for fname in os.listdir(left_mask_dir):
    if fname.endswith(".png"):
        left = Image.open(os.path.join(left_mask_dir, fname)).convert("L")
        right = Image.open(os.path.join(right_mask_dir, fname)).convert("L")

        left = np.array(left)
        right = np.array(right)
        combined = np.clip(left + right, 0, 255)  # Combine both lungs

        out = Image.fromarray(combined.astype(np.uint8))
        out.save(os.path.join(output_mask_dir, fname))
