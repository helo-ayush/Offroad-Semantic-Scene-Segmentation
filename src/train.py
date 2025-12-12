import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import DualityDataset
from src.model_refine import ProgressiveSemanticSegmenter # NEW Model
from src.utils import CLASS_DEFINITIONS, ID_TO_NAME
import os
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def train_model(epochs=15, batch_size=4, lr=1e-4, device='cuda'):
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
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=512),
        A.RandomCrop(height=input_h, width=input_w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
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
    class_weights = torch.tensor([0.047, 0.334, 0.324, 0.251, 1.562, 0.801, 2.464, 0.731, 0.146, 0.073]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

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
    train_model(epochs=15, batch_size=4, device=device)

