import os
import shutil
import random
from pathlib import Path

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
    Split BUSI dataset into train/val/test sets while preserving original filenames
    and balancing the number of samples per class (benign/malignant).
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    random.seed(seed)

    classes = ["benign", "malignant"]
    splits = ["train", "val", "test"]

    # Clean and recreate output directories
    for split in splits:
        for subfolder in ["images", "masks"]:
            path = Path(output_dir) / split / subfolder
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_path = Path(base_dir) / cls
        image_files = sorted([f for f in cls_path.glob("*.png") if "_mask" not in f.name])
        mask_files = sorted([f for f in cls_path.glob("*.png") if "_mask" in f.name])

        paired = []
        for img_path in image_files:
            mask_path = cls_path / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                paired.append((img_path, mask_path))

        random.shuffle(paired)
        n_total = len(paired)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        splits_data = {
            "train": paired[:n_train],
            "val": paired[n_train:n_train + n_val],
            "test": paired[n_train + n_val:]
        }

        for split, samples in splits_data.items():
            for img_path, mask_path in samples:
                # Preserve original filenames
                shutil.copy(img_path, Path(output_dir) / split / "images" / img_path.name)
                shutil.copy(mask_path, Path(output_dir) / split / "masks" / mask_path.name)

    print("✅ Balanced split complete with original filenames preserved.")

split_busi_dataset_balanced()
