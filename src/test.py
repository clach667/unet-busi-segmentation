import torch
from torch.utils.data import DataLoader
from dataset import BUSIDataset
from unet import BetterUNet
import torch.nn.functional as F

import os


from torchvision.utils import save_image

# ====== Params ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
MODEL_PATH = "/Users/clarachoukroun/unet-busi-project/results/unet_final.pt"
IMAGE_SIZE = (256, 256)
SAVE_DIR = "results/test_preds"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== Dice Score ======
def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    targets = targets.bool()
    intersection = (preds & targets).float().sum((1,2,3))
    union = preds.float().sum((1,2,3)) + targets.float().sum((1,2,3))
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.mean().item()

# ====== Accuracy ======
def binary_accuracy(preds, targets, threshold=0.5):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    return (preds_bin == targets).float().mean().item()

# ====== Dataset & Loader ======
dataset = BUSIDataset("/Users/clarachoukroun/unet-busi-project/data/images", "/Users/clarachoukroun/unet-busi-project/data/masks", size=IMAGE_SIZE)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
_, _, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# ====== Load Model ======
model = BetterUNet(init_features=32).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ====== Evaluation ======
total_dice, total_acc = 0, 0
print("Evaluating on test set...")

with torch.no_grad():
    for i, (imgs, masks) in enumerate(test_loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)

        total_dice += dice_score(preds, masks)
        total_acc += binary_accuracy(preds, masks)

        # Sauvegarder les prédictions visuelles
        out = torch.sigmoid(preds)
        save_image(out, f"{SAVE_DIR}/pred_{i}.png")
        save_image(masks, f"{SAVE_DIR}/mask_{i}.png")

n = len(test_loader)
print(f"Test Dice Score: {total_dice / n:.4f}")
print(f"Test Accuracy: {total_acc / n:.4f}")
