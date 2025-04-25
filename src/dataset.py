import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from torchvision import transforms

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256), augment_malignant=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment_malignant = augment_malignant

        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png") and "_mask" not in f
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        mask_name = image_name.replace(".png", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.augment_malignant and "malignant" in image_name.lower():
            image = self.aug_transform(image)
            mask = self.aug_transform(mask)
        else:
            image = self.base_transform(image)
            mask = self.base_transform(mask)

        return image, mask
