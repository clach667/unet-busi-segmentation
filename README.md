# 🧠 Breast Ultrasound Segmentation (U-Net)

This project implements a lightweight U-Net model for segmenting breast lesions in ultrasound images (BUSI dataset).

## 🔍 Project Overview
- Input: grayscale ultrasound images
- Output: binary segmentation masks
- Model: Custom lightweight U-Net
- Loss: BCE + Dice Loss
- Framework: PyTorch

## 📊 Results
- **Dice Score**: 0.5055
- **Accuracy**: 91.76%
- See examples in `results/test_preds/`

## 🧪 Evaluation examples
![Example 1](figures/pred1.png)
![Example 2](figures/pred2.png)

## 🚀 Usage
```bash
python train.py
python test.py
