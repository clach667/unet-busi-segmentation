import os
import shutil

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

if __name__ == "__main__":
    prepare_busi_dataset()
