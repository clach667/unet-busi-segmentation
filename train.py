import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,  WeightedRandomSampler
from tqdm import tqdm
from src.dataset import BUSIDataset
from src.unet import UNet  # Tu peux aussi l'appeler BetterUNet si tu veux
from src.utils import dice_loss, binary_accuracy
from src.early_stopping import EarlyStopping

# ====== Config ======
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Dataset ======
from torch.utils.data import DataLoader, WeightedRandomSampler

train_set = BUSIDataset("data/train/images", "data/train/masks", size=(256, 256), augment_malignant=True)
val_set = BUSIDataset("data/val/images", "data/val/masks", size=(256, 256), augment_malignant=False)

class_counts = [0, 0]
for fname in train_set.image_names:
    label = 0 if "benign" in fname.lower() else 1
    class_counts[label] += 1

weights = [1.0 / class_counts[0] if "benign" in fname.lower() else 1.0 / class_counts[1] for fname in train_set.image_names]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


train_loader = DataLoader(train_set, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=8)

# ====== Model ======
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ====== Logging ======
train_logs = {"loss": [], "acc": []}
val_logs = {"loss": [], "acc": []}

# ====== Early Stopping ======
early_stopper = EarlyStopping(patience=3, delta=0.01, path="results/unet_final.pt")

# ====== Training Loop ======
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_acc = 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for imgs, masks in loop:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        bce = F.binary_cross_entropy_with_logits(preds, masks)
        dice = dice_loss(preds, masks)
        loss = bce + dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += binary_accuracy(preds, masks)
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    train_logs["loss"].append(avg_loss)
    train_logs["acc"].append(avg_acc)

    # ====== Validation ======
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            bce = F.binary_cross_entropy_with_logits(preds, masks)
            dice = dice_loss(preds, masks)
            val_loss += (bce + dice).item()
            val_acc += binary_accuracy(preds, masks)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_logs["loss"].append(val_loss)
    val_logs["acc"].append(val_acc)

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

