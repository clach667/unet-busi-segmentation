# Breast Ultrasound Segmentation (U-Net)

This project implements a lightweight U-Net model to segment breast tumors in ultrasound images, based on the **BUSI dataset**. The goal is to detect and segment lesions from grayscale ultrasound scans using deep learning.

## Dataset
The original dataset contains three classes: `benign`, `malignant`, and `normal`. Only the `benign` and `malignant` samples are used for segmentation, as the `normal` class does not contain lesions.

### Dataset Preprocessing
The dataset.py file contains a custom Dataset for BUSI (BUSIDataset) :
1. Defines a PyTorch Dataset class for loading grayscale breast ultrasound images and their corresponding masks.
2. Automatically matches each image with its _mask.png file based on filename.
3. Applies basic preprocessing: resizing and tensor conversion.
4. Includes optional data augmentation (horizontal flip and rotation) for malignant cases to improve robustness during training.
5. Returns (image, mask) pairs ready for training segmentation models.

The split_busi_dataset_balanced.py ensures a balanced number of benign and malignant samples in each split.
1. Splits the original BUSI dataset into train, validation, and test sets using a configurable ratio (default: 80/10/10).
2. Preserves original filenames for traceability.
3. Automatically creates a clean directory structure:
 data/
  ├── train/
  │   ├── images/
  │   └── masks/
  ├── val/
  │   ├── images/
  │   └── masks/
  └── test/
      ├── images/
      └── masks/

## Model: UNet
This project uses a UNet convolutional neural network architecture, specifically designed for biomedical image segmentation. The model is trained to detect and segment lesion regions in breast ultrasound images from the BUSI dataset.

### Architecture Details
The UNet architecture follows a symmetric encoder-decoder design with skip connections. 
1. The encoder (Contracting Path) is composed of several blocks of two 3×3 convolutions followed by a ReLU activation and a 2×2 max pooling. The number of feature channels is doubled at each step to capture more abstract features.
2. The bottleneck connects the encoder and decoder. It captures the most abstract representation of the image before upsampling.
3. The decoder (Expanding Path):is composed of blocks of transposed convolution, followed by two 3×3 convolutions and ReLU.
4. The output layer is a final 1×1 convolution that reduces the number of channels to 1 an a sigmoid activation is applied to generate a binary mask (pixel-wise classification).

### Loss Function
The model uses two loss functions ; Binary Cross Entropy loss (BCE) that measures the per-pixel error between predictions and ground truth masks and Dice loss that measures the overlap between predicted and true regions
The total training loss is the sum of both: BCE + Dice.

### Hyperparameters
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
