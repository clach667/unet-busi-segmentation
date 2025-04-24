# Breast Ultrasound Segmentation (U-Net)

This project implements a lightweight U-Net model to segment breast tumors in ultrasound images, based on the **BUSI dataset**. The goal is to detect and segment lesions from grayscale ultrasound scans using deep learning.

---

## Dataset

The original dataset contains three classes: `benign`, `malignant`, and `normal`. Only the `benign` and `malignant` samples are used for segmentation, as the `normal` class does not contain lesions.

### Dataset Preprocessing
- All image/mask pairs were extracted from the original folders.
- Only samples with both an image and a corresponding `_mask.png` file were included.
- A **balanced split** was performed per class into:
  - 80% training
  - 10% validation
  - 10% testing
- All images and masks were resized to **256×256**.
- Final structure:
`data/ ├── train/ │ ├── images/ │ └── masks/ ├── val/ │ ├── images/ │ └── masks/ └── test/ ├── images/ └── masks/`

## Model: BetterUNet

A lightweight U-Net with:
- Reduced number of filters
- `Dropout2d` for regularization
- Final sigmoid activation for binary segmentation

Trained using:
- **Binary Cross Entropy** + **Dice Loss**
- **Adam optimizer**, learning rate: `1e-4`
- Early stopping on validation loss

---

## Training

```bash
python train.py
```
Training logs (loss and accuracy) are saved in results/ as .npy files for plotting.

## Evaluation

```bash
python test.py
```
This script:

Loads the trained model (results/unet_final.pt)
Evaluates on the test set
Computes average Dice score and accuracy
Saves predictions in results/test_preds/

## Visualization 

In the notebook notebooks/ProjectSummary.ipynb, you can:

Display the model architecture
Visualize predictions alongside ground truths
Plot training/validation loss curves
Summarize final performance on unseen data