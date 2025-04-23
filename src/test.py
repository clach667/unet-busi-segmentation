from dataset import BUSIDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

dataset = BUSIDataset("data/images", "data/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# On récupère un batch
images, masks = next(iter(loader))

# Affiche les 4 premières images avec leurs masques
for i in range(4):
    img = images[i][0]  # channel 0 (grayscale)
    mask = masks[i][0]

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(img, cmap="gray")
    plt.imshow(mask, cmap="jet", alpha=0.4)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
