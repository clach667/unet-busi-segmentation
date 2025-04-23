import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from torchvision import transforms

class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize(self.size),   # Resize PIL image first
            transforms.ToTensor()           # Then convert to [0,1] float
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0).float()  # Binarization step

        return image, mask

