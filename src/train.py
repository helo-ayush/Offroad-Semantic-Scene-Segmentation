import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import DualityDataset
from src.model_refine import ProgressiveSemanticSegmenter # NEW Model
from src.loss import CombinedLoss
from src.utils import CLASS_DEFINITIONS, ID_TO_NAME
import os
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def train_model(epochs=10, batch_size=4, lr=1e-4, device='cuda'):
    # Config
    config = load_config()
    input_h, input_w = config["input_size"]
    
    # Paths
    data_dir = "Offroad_Segmentation_Training_Dataset"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Augmentations
    # Augmentations
    # Improved pipeline: Maintain higher res details using RandomCrop
    # Added robustness for offroad conditions (lighting, geometry)
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=512),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.5), # Rotation/Scale
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0) # alpha_affine removed in v2
        ], p=0.3),
        A.RandomCrop(height=input_h, width=input_w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5), # Lighting
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3), # Color shift
        # V2 API: num_holes_range, hole_height_range, hole_width_range
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), p=0.3), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(input_h, input_w), # Validation still uses full image context
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Dataset & Loader
    train_dataset = DualityDataset(data_dir, split="train", transform=train_transform)
    val_dataset = DualityDataset(data_dir, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize New Refined Model
    print(">> Initializing PSD-Net (Progressive Semantic Decoder)...")
    model = ProgressiveSemanticSegmenter(n_classes=len(CLASS_DEFINITIONS)) 
    model.to(device)
    
    # Clean Slate Training (No Checkpoint Loading)
    # checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    # if os.path.exists(checkpoint_path):
    #     print(f">> Loading existing checkpoint from {checkpoint_path} for fine-tuning...")
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Optimization
    # Calculated Inverse Frequency Weights to handle class imbalance
    # [Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky]
    class_weights = torch.tensor([0.0022, 0.1121, 0.1054, 0.0632, 2.4415, 0.6423, 6.0718, 0.5346, 0.0215, 0.0054]).to(device)
    
    # Using Combined Focal + Dice Loss for better class balance handling
    criterion = CombinedLoss(alpha=class_weights, gamma=2.0, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_val_loss = float('inf')

    print(f"Training on {device} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).long()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(">> Saved Best Model")

    print("Training Complete")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DINOv2 is VRAM hungry, start with small batch
    train_model(epochs=5, batch_size=4, device=device)

