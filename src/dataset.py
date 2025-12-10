import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.utils import map_mask_values


class DualityDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Structure: root_dir/train/Color_Images/*.jpg
        self.split_dir = os.path.join(self.root_dir, split)
        self.images_dir = os.path.join(self.split_dir, "Color_Images")
        self.masks_dir = os.path.join(self.split_dir, "Segmentation")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Missing images at: {self.images_dir}")
            
        self.image_filenames = sorted(os.listdir(self.images_dir))
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load Image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load corresponding mask
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        mask_pil = Image.open(mask_path)
        mask_np = np.array(mask_pil)
        
        # Map raw mask values to class IDs
        mask = map_mask_values(mask_np)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.long()
