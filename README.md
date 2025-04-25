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

## Model: UNet
This project uses a UNet convolutional neural network architecture, specifically designed for biomedical image segmentation. The model is trained to detect and segment lesion regions in breast ultrasound images from the BUSI dataset.

# Architecture Details
The UNet architecture follows a symmetric encoder-decoder design with skip connections. 
1. The encoder (Contracting Path) is composed of several blocks of two 3×3 convolutions followed by a ReLU activation and a 2×2 max pooling. The number of feature channels is doubled at each step to capture more abstract features.
2. The bottleneck connects the encoder and decoder. It captures the most abstract representation of the image before upsampling.
3. The decoder (Expanding Path):is composed of blocks of transposed convolution, followed by two 3×3 convolutions and ReLU.
4. The output layer is a final 1×1 convolution that reduces the number of channels to 1 an a sigmoid activation is applied to generate a binary mask (pixel-wise classification).

# Loss Function
The model uses two loss functions ; Binary Cross Entropy loss (BCE) that measures the per-pixel error between predictions and ground truth masks and Dice loss that measures the overlap between predicted and true regions
The total training loss is the sum of both: BCE + Dice.

# Hyperparameters
Optimizer: Adam 
Learning rate: 1e-4.

## Training
To correct class imbalance, we apply a targeted data augmentation to malignant tumor images that are underrepresented. We also apply a WeightedRandomSampler ensures a balanced contribution from benign and malignant samples. There's also a possible EarlyStopping if the validation doesn't improve fast enough. Training logs (loss and accuracy) are saved in results/ as .npy files for plotting. Metrics tracked during training: Total Loss and Binary Accuracy.

```bash
python train.py
```

## Evaluation
This script:

Loads the trained model (results/unet_final.pt)
Evaluates on the test set
Computes average Dice score and accuracy
Saves predictions in results/test_preds/

```bash
python test.py
```


## Visualization 

In the notebook notebooks/ProjectSummary.ipynb, you can:

Display the model architecture
Visualize predictions alongside ground truths
Plot training/validation loss curves
Summarize final performance on unseen data
