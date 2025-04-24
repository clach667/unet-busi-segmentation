# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from src.dataset import BUSIDataset
from src.unet import BetterUNet
from src.utils import dice_loss, binary_accuracy

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4  # Lower learning rate
PATIENCE = 5

# Data
train_set = BUSIDataset("data/train/images", "data/train/masks")
val_set = BUSIDataset("data/val/images", "data/val/masks")
test_set = BUSIDataset("data/test/images", "data/test/masks")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=1)

# Model
model = BetterUNet(init_features=32, dropout=0.2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Logs
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Early Stopping
best_val_loss = float('inf')
patience_counter = 0

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

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    train_losses.append(avg_loss)
    val_losses.append(val_loss)
    train_accs.append(avg_acc)
    val_accs.append(val_acc)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "results/unet_final.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Save training logs
np.save("results/train_losses.npy", np.array(train_losses))
np.save("results/val_losses.npy", np.array(val_losses))
np.save("results/train_accs.npy", np.array(train_accs))
np.save("results/val_accs.npy", np.array(val_accs))
