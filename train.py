import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BUSIDataset
from src.unet import MiniUNet

from torchsummary import summary

# Set hyperparameters
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
train_dataset = BUSIDataset("data/images", "data/masks", size=IMAGE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize U-Net model
model = MiniUNet().to(DEVICE)

# Print model summary
summary(model, input_size=(1, IMAGE_SIZE[0], IMAGE_SIZE[1]))

# Set loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    print("Training started")
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)

    for images, masks in loop:
        torch.save(model.state_dict(), f"results/unet_epoch{epoch+1}.pt")

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss / len(train_loader):.4f}")

