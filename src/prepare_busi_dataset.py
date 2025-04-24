import os
import shutil
import random

def prepare_busi_dataset(base_dir="data/Dataset_BUSI_with_GT"):
    output_images = "data/images"
    output_masks = "data/masks"
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    for class_name in ["benign", "malignant"]:
        class_path = os.path.join(base_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(".png") and "_mask" not in filename:
                image_path = os.path.join(class_path, filename)
                mask_path = os.path.join(class_path, filename.replace(".png", "_mask.png"))

                if os.path.exists(mask_path):
                    shutil.copy(image_path, os.path.join(output_images, filename))
                    shutil.copy(mask_path, os.path.join(output_masks, filename))
                    print(f"✔ Copied: {filename} and its mask.")
                else:
                    print(f"⚠ No mask found for: {filename}")

def split_dataset(image_dir, mask_dir, output_dir="data", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset into train, val, and test sets.
    
    Parameters:
        image_dir (str): Path to the original images.
        mask_dir (str): Path to the corresponding masks.
        output_dir (str): Base directory for the split data.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        seed (int): Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1."

    random.seed(seed)
    all_images = sorted(os.listdir(image_dir))
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:]
    }

    for split_name, files in splits.items():
        img_split_dir = os.path.join(output_dir, split_name, "images")
        mask_split_dir = os.path.join(output_dir, split_name, "masks")
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(mask_split_dir, exist_ok=True)

        for file in files:
            shutil.copy(os.path.join(image_dir, file), os.path.join(img_split_dir, file))
            shutil.copy(os.path.join(mask_dir, file), os.path.join(mask_split_dir, file))

    print("Dataset split completed.")

if __name__ == "__main__":
    split_dataset("/Users/clarachoukroun/unet-busi-project/data/images","/Users/clarachoukroun/unet-busi-project/data/masks")
