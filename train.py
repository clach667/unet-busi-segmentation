import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from src.dataset import BUSIDataset
from src.unet import BetterUNet
from tqdm import tqdm
from torchsummary import summary

# ====== Params ======
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Dataset & Splits ======
dataset = BUSIDataset("/Users/clarachoukroun/unet-busi-project/data/images", "/Users/clarachoukroun/unet-busi-project/data/masks", size=(256, 256))
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ====== Dice Loss ======
def dice_loss(preds, targets, smooth=1e-8):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# ====== Accuracy ======
def binary_accuracy(preds, targets, threshold=0.5):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    return (preds_bin == targets).float().mean().item()

# ====== Model & Optim ======
model = BetterUNet(init_features=32).to(DEVICE)
summary(model, input_size=(1, 256, 256))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc / len(val_loader):.4f}")

# Save the final model
torch.save(model.state_dict(), "results/unet_final.pt")
