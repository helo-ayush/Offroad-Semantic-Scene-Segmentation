import numpy as np
import torch
import matplotlib.pyplot as plt

# 10-Class Offroad Mapping
CLASS_DEFINITIONS = [
    {"name": "Background",      "id": 0, "value": 0,     "color": [0, 0, 0]},
    {"name": "Trees",           "id": 1, "value": 100,   "color": [0, 100, 0]},
    {"name": "Lush Bushes",     "id": 2, "value": 200,   "color": [50, 205, 50]},
    {"name": "Dry Grass",       "id": 3, "value": 300,   "color": [240, 230, 140]},
    {"name": "Dry Bushes",      "id": 4, "value": 500,   "color": [189, 183, 107]},
    {"name": "Ground Clutter",  "id": 5, "value": 550,   "color": [128, 0, 0]},
    {"name": "Logs",            "id": 6, "value": 700,   "color": [139, 69, 19]},
    {"name": "Rocks",           "id": 7, "value": 800,   "color": [128, 128, 128]},
    {"name": "Landscape",       "id": 8, "value": 7100,  "color": [47, 79, 79]},
    {"name": "Sky",             "id": 9, "value": 10000, "color": [135, 206, 235]}
]

VALUE_TO_ID = {c["value"]: c["id"] for c in CLASS_DEFINITIONS}

def map_mask_values(mask_np):
    # Vectorized mapping of raw values to class IDs
    output = np.zeros_like(mask_np, dtype=np.int64)
    for c in CLASS_DEFINITIONS:
        output[mask_np == c["value"]] = c["id"]
    return output

# Decoding params
IDX_TO_COLOR = [c["color"] for c in CLASS_DEFINITIONS]
ID_TO_IDX = {c["id"]: i for i, c in enumerate(CLASS_DEFINITIONS)}
ID_TO_NAME = {c["id"]: c["name"] for c in CLASS_DEFINITIONS}

def decode_segmap(image, num_classes=len(CLASS_DEFINITIONS)):
    # Converts class ID map to RGB image for visualization
    label_colors = np.array(IDX_TO_COLOR).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def show_img_target(img, target, pred=None):
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).cpu().numpy()
        
    if torch.is_tensor(target):
        target = target.cpu().numpy()
        
    target_rgb = decode_segmap(target)
    
    if pred is not None:
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        pred_rgb = decode_segmap(pred)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].set_title("Image")
        ax[1].imshow(target_rgb)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred_rgb)
        ax[2].set_title("Prediction")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Image")
        ax[1].imshow(target_rgb)
        ax[1].set_title("Ground Truth")
    
    plt.show()
