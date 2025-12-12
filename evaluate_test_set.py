import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.model_refine import ProgressiveSemanticSegmenter
from src.utils import CLASS_DEFINITIONS, map_mask_values
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def evaluate():
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    input_size = (252, 252)
    checkpoint_path = "checkpoints/best_model.pth"
    
    test_dir = "Offroad_Segmentation_testImages"
    images_dir = os.path.join(test_dir, "Color_Images")
    masks_dir = os.path.join(test_dir, "Segmentation")
    
    output_report = "Test_Optimization_Report_New.txt"
    
    # Model
    print(f"Loading Model on {device}...")
    model = ProgressiveSemanticSegmenter(n_classes=len(CLASS_DEFINITIONS))
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(">> Loaded checkpoint.")
    else:
        print("!! Checkpoint not found. Running with random weights (for testing only).")
    
    model.to(device)
    model.eval()
    
    # Transform
    transform = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # List images
    if not os.path.exists(images_dir):
        print(f"Error: Test directory not found at {images_dir}")
        return

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"Evaluating on {len(image_files)} images...")
    
    results = []
    
    with open(output_report, "w") as f:
        f.write("Evaluation of PSD-Net (Progressive Semantic Decoder)\n")
        f.write("--------------------------------------------------\n\n")
        
        for img_name in tqdm(image_files):
            img_path = os.path.join(images_dir, img_name)
            
            # Load Image
            image = np.array(Image.open(img_path).convert("RGB"))
            
            # Load Mask
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_path = os.path.join(masks_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
                
            mask_pil = Image.open(mask_path)
            mask_raw = np.array(mask_pil)
            mask_gt = map_mask_values(mask_raw)
            
            # Inference
            augmented = transform(image=image)
            input_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Resize prediction to original mask size for evaluation
            # (Or resize mask to prediction size? Usually we evaluate at original resolution if possible, 
            # but our model outputs fixed size 252x252. Let's resize mask to 252x252 for consistency with training metrics)
            h, w = prediction.shape
            mask_resized = cv2.resize(mask_gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Pixel Accuracy
            correct = (prediction == mask_resized).sum()
            total = prediction.size
            acc = (correct / total) * 100
            
            log_line = f"Image: {img_name} | Accuracy: {acc:.2f}%"
            results.append(acc)
            f.write(log_line + "\n")
            
        avg_acc = sum(results) / len(results) if results else 0
        summary = f"\nAverage Pixel Accuracy: {avg_acc:.2f}%"
        print(summary)
        f.write(summary + "\n")

    print(f"Report saved to {output_report}")

if __name__ == "__main__":
    evaluate()
