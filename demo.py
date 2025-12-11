import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from PIL import Image
from src.model_refine import ProgressiveSemanticSegmenter as UNet # Alias for compatibility
from src.utils import CLASS_DEFINITIONS
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/best_model.pth"
INPUT_SIZE = (252, 252)
LABELS = [c["name"] for c in CLASS_DEFINITIONS]
COLORS = [np.array(c["color"]) / 255.0 for c in CLASS_DEFINITIONS]

class OffRoadDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offroad Semantic Scene Segmentation")
        self.root.state('zoomed') # Fullscreen window
        
        # Backend Init
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.transform = A.Compose([
            A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # GUI Layout
        self.setup_ui()
        
        # Plotting Setup
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_img = None
        self.ax_pred = None
        self.ax_legend = None
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Show initial welcome message on plot
        self.show_welcome()

    def load_model(self):
        print(f"Loading Model on {self.device}...")
        # ProgressiveSemanticSegmenter only needs n_classes
        model = UNet(n_classes=len(CLASS_DEFINITIONS))
        if os.path.exists(CHECKPOINT_PATH):
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
        else:
            messagebox.showerror("Error", "Checkpoint not found!")
            exit()
        model.to(self.device)
        model.eval()
        return model

    def setup_ui(self):
        # -- Top Bar --
        self.top_frame = tk.Frame(self.root, bg="#333333", pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_select = tk.Button(self.top_frame, text="SELECT IMAGE (O)", command=self.on_select_image,
                                    font=("Segoe UI", 12, "bold"), bg="#4CAF50", fg="white", 
                                    activebackground="#45a049", padx=20)
        self.btn_select.pack(side=tk.LEFT, padx=20)
        
        self.lbl_status = tk.Label(self.top_frame, text=f"System Ready | Device: {self.device.upper()}",
                                   font=("Consolas", 10), bg="#333333", fg="#AAAAAA")
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        # -- Main Area --
        self.plot_frame = tk.Frame(self.root, bg="white")
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Keyboard Shortcut
        self.root.bind('<o>', lambda e: self.on_select_image())

    def show_welcome(self):
        self.fig.clear()
        self.fig.text(0.5, 0.5, "DUALITY AI SYSTEM\n\nClick 'SELECT IMAGE' to Start Analysis", 
                      ha='center', va='center', fontsize=20, color='gray')
        self.canvas.draw()

    def on_select_image(self):
        initial_dir = os.path.abspath("Offroad_Segmentation_testImages/Color_Images")
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
            
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Test Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, img_path):
        self.lbl_status.config(text="Processing...", fg="yellow")
        self.root.update()
        
        try:
            # Load Image
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img)
            
            # Auto-detect Ground Truth based on directory
            mask_np = None
            parent_dir = os.path.dirname(img_path)
            if "Color_Images" in parent_dir:
                mask_dir = parent_dir.replace("Color_Images", "Segmentation")
                basename = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(mask_dir, basename + ".png")
                
                if os.path.exists(mask_path):
                    from src.utils import map_mask_values
                    mask_pil = Image.open(mask_path)
                    mask_raw = np.array(mask_pil)
                    mask_np = map_mask_values(mask_raw)
                    print(f"Loaded GT: {mask_path}")

            # Preprocess & Inference
            augmented = self.transform(image=img_np)
            input_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
                confidence = torch.max(probabilities, dim=1)[0].squeeze(0).cpu().numpy()
                mean_conf = np.mean(confidence) * 100
                
            inf_time = (time.time() - start_time) * 1000
            
            # Metrics
            accuracy_text = "N/A"
            if mask_np is not None:
                # Resize GT to match prediction (252x252) for fair acc calculation
                import cv2
                h, w = prediction.shape
                mask_resized = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                
                correct = (prediction == mask_resized).sum()
                acc = (correct / prediction.size) * 100
                accuracy_text = f"{acc:.1f}%"

            # Update GUI
            status_msg = f"Done | Time: {inf_time:.1f}ms | Conf: {mean_conf:.1f}%"
            if mask_np is not None:
                status_msg += f" | Accuracy: {accuracy_text}"
            
            self.lbl_status.config(text=status_msg, fg="#4CAF50")
            
            self.update_plot(pil_img, prediction, mask_np, mean_conf, inf_time, accuracy_text)
            
        except Exception as e:
            print(e)
            self.lbl_status.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", str(e))

    def update_plot(self, original_img, prediction, mask_np, mean_conf, inf_time, acc_text):
        self.fig.clear()
        import cv2
        
        # Original Dimensions (PIL is W, H)
        w_orig, h_orig = original_img.size
        
        # Resize Prediction to match Original Image for visualization
        # Prediction is currently 252x252 (Model Output)
        prediction_resized = cv2.resize(prediction.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # Adjust grid based on whether we have Ground Truth
        if mask_np is not None:
             gs = self.fig.add_gridspec(2, 4) # 3 columns for images, 1 for legend
        else:
             gs = self.fig.add_gridspec(2, 3)

        # 1. Input
        ax1 = self.fig.add_subplot(gs[0:2, 0])
        ax1.imshow(original_img)
        ax1.set_title("Input Feed", fontsize=10, fontweight='bold')
        ax1.axis('off')

        # 2. Prediction (Resized)
        ax2 = self.fig.add_subplot(gs[0:2, 1])
        # Create RGB map
        rgb = np.zeros((h_orig, w_orig, 3))
        for i, color in enumerate(COLORS):
            rgb[prediction_resized == i] = color
            
        ax2.imshow(rgb)
        ax2.set_title(f"AI Prediction\n(Conf: {mean_conf:.1f}%)", fontsize=10, fontweight='bold')
        ax2.axis('off')

        next_col = 2
        
        # 3. Ground Truth (Optional)
        if mask_np is not None:
            ax3 = self.fig.add_subplot(gs[0:2, 2])
            h_m, w_m = mask_np.shape
            rgb_m = np.zeros((h_m, w_m, 3))
            for i, color in enumerate(COLORS):
                # Mask might need resizing if it wasn't original size? 
                # Currently mask_np is loaded from file, so it should match original_img size approximately
                # But let's be safe and resize visual mask too if needed, or just assume it matches
                # If mask_np shape differs from rgb shape, imshow might handle it or we resize
                if mask_np.shape != (h_orig, w_orig):
                     mask_to_show = cv2.resize(mask_np.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                else:
                     mask_to_show = mask_np
                
                rgb_m[mask_to_show == i] = color
                
            ax3.imshow(rgb_m)
            ax3.set_title(f"Ground Truth\n(Acc: {acc_text})", fontsize=10, fontweight='bold')
            ax3.axis('off')
            next_col = 3

        # 4. Stats & Legend
        ax_leg = self.fig.add_subplot(gs[0:2, next_col])
        ax_leg.axis('off')
        
        # Stats
        stats = (
            f"METRICS\n"
            f"-------\n"
            f"Latency : {inf_time:.0f} ms\n"
            f"Conf    : {mean_conf:.0f} %\n"
            f"Accuracy: {acc_text}\n"
        )
        ax_leg.text(0.05, 0.95, stats, transform=ax_leg.transAxes, fontsize=11, family='monospace', va='top')
        
        # Legend
        legend_patches = [mpatches.Patch(color=COLORS[i], label=lab) for i, lab in enumerate(LABELS)]
        ax_leg.legend(handles=legend_patches, loc='center', title="Classes", frameon=False, fontsize=9)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OffRoadDemoApp(root)
    root.mainloop()
