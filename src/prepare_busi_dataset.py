import os
import shutil
import random
from pathlib import Path

def split_busi_dataset_balanced(
    base_dir="data/Dataset_BUSI_with_GT",
    output_dir="data",
    split_ratio=(0.8, 0.1, 0.1),
    seed=42
):
    """
    Split the BUSI dataset into train/val/test sets in a balanced way across 'benign' and 'malignant' classes.
    
    Parameters:
        base_dir (str): Path to the original BUSI dataset (with 'benign' and 'malignant' subfolders).
        output_dir (str): Root output directory for the split dataset.
        split_ratio (tuple): Train/val/test ratio (should sum to 1.0).
        seed (int): Random seed for reproducibility.
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    random.seed(seed)
    
    classes = ["benign", "malignant"]
    splits = ["train", "val", "test"]

    # Clean existing output folders
    for split in splits:
        for sub in ["images", "masks"]:
            split_path = Path(output_dir) / split / sub
            if split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)

    # Process each class separately
    for cls in classes:
        cls_path = Path(base_dir) / cls
        image_files = sorted([f for f in cls_path.glob("*.png") if "mask" not in f.name])
        mask_files = sorted([f for f in cls_path.glob("*.png") if "mask" in f.name])

        paired = list(zip(image_files, mask_files))
        random.shuffle(paired)

        n = len(paired)
        n_train = int(split_ratio[0] * n)
        n_val = int(split_ratio[1] * n)

        split_data = {
            "train": paired[:n_train],
            "val": paired[n_train:n_train + n_val],
            "test": paired[n_train + n_val:]
        }

        for split in splits:
            for img_path, mask_path in split_data[split]:
                shutil.copy(img_path, Path(output_dir) / split / "images" / img_path.name)
                shutil.copy(mask_path, Path(output_dir) / split / "masks" / mask_path.name)

    print("✅ Balanced dataset split completed.")

# Usage:
# split_busi_dataset_balanced()


if __name__ == "__main__":
    split_busi_dataset_balanced("/Users/clarachoukroun/unet-busi-project/data/images","/Users/clarachoukroun/unet-busi-project/data/masks")
