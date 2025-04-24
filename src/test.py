import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import sys

from unet import UNet
import os
sys.path.append(os.path.abspath(".."))

from dataset import BUSIDataset

from utils import compute_dice, compute_accuracy

# ----------- Config ---------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/unet_final.pt"  
IMAGE_SIZE = (256, 256)

# ----------- Dataset -------------
test_set = BUSIDataset("data/test/images", "data/test/masks", size=IMAGE_SIZE)
test_loader = DataLoader(test_set, batch_size=1)

# ----------- Load model ----------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------- Evaluation ----------
dice_scores = []
acc_scores = []
n_visuals = 4
shown = 0


with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = torch.sigmoid(model(img))

            dice = compute_dice(pred, mask)
            acc = compute_accuracy(pred, mask)
            dice_scores.append(dice.item())
            acc_scores.append(acc.item())

            if shown < n_visuals:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img[0].cpu().squeeze(), cmap="gray")
                axs[0].set_title("Image")
                axs[1].imshow(mask[0].cpu().squeeze(), cmap="gray")
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred[0].cpu().squeeze(), cmap="viridis")
                axs[2].set_title("Prediction (sigmoid)")
                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                shown += 1    

# ----------- Results ------------
print(f"Test Dice Score: {np.mean(dice_scores):.4f}")
print(f"Test Accuracy:   {np.mean(acc_scores):.4f}")
