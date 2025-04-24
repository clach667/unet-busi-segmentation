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
    Split BUSI dataset into train/val/test, balanced per class, and save with matching filenames.
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    random.seed(seed)
    
    classes = ["benign", "malignant"]
    splits = ["train", "val", "test"]

    # Reset output directories
    for split in splits:
        for sub in ["images", "masks"]:
            split_path = Path(output_dir) / split / sub
            if split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)

    counter = {split: 0 for split in splits}

    for cls in classes:
        cls_path = Path(base_dir) / cls
        image_files = sorted([f for f in cls_path.glob("*.png") if "_mask" not in f.name])
        mask_files = sorted([f for f in cls_path.glob("*.png") if "_mask" in f.name])

        paired = []
        for img in image_files:
            mask_name = img.stem + "_mask.png"
            mask_path = cls_path / mask_name
            if mask_path.exists():
                paired.append((img, mask_path))

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
                idx = counter[split]
                name = f"{split}_{idx:04d}.png"

                shutil.copy(img_path, Path(output_dir) / split / "images" / name)
                shutil.copy(mask_path, Path(output_dir) / split / "masks" / name)

                counter[split] += 1

    print("✅ Balanced split complete with matching filenames.")


if __name__ == "__main__":
    split_busi_dataset_balanced()
