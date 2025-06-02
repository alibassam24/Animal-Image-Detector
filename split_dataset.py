import os
import shutil
import random
from pathlib import Path

# ✅ Automatically get current project directory (where the script is)
base_dir = Path(__file__).resolve().parent
raw_dir = base_dir / "archive" / "raw-img"
output_dir = base_dir / "data"

train_dir = output_dir / "train"
val_dir = output_dir / "val"

# ✅ Create output folders if they don’t exist
for folder in [train_dir, val_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# ✅ Set seed for reproducibility
random.seed(42)

# ✅ Loop through each class folder in raw-img
for class_folder in raw_dir.iterdir():
    if class_folder.is_dir():
        images = list(class_folder.glob("*.*"))  # All image files
        random.shuffle(images)

        # ✅ Split into train and val (80-20)
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # ✅ Create subfolders for class
        (train_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_folder.name).mkdir(parents=True, exist_ok=True)

        # ✅ Copy training images
        for img in train_images:
            shutil.copy(img, train_dir / class_folder.name / img.name)

        # ✅ Copy validation images
        for img in val_images:
            shutil.copy(img, val_dir / class_folder.name / img.name)

print("✅ Dataset split into 'data/train/' and 'data/val/' successfully.")
