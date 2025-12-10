# Offroad Semantic Scene Segmentation 🚙💨

> **Hackathon Submission 2024**  
> *Robust Terrain Analysis using DINOv2 Backbone & U-Net*

![Project Banner](Visual_Report/Visual_Report_1_0000612.png)

## 📌 Overview
Self-driving cars are great on highways, but off-road environments are a nightmare. There are no lane markings, no stop signs, and "drivable path" is distinct from "tall grass" or "mud". 

**Offroad Semantic Scene Segmentation** is a system designed specifically for Unmanned Ground Vehicles (UGVs) operating in unstructured environments. It helps robots distinguish between **safe terrain** (dry grass, dirt) and **obstacles** (rocks, trees, deep water).

## 🧠 Model Architecture: The "Vision Transformer" Advantage

We didn't just train a standard U-Net. We used **Transfer Learning** with a state-of-the-art backbone.

### 1. The Backbone: DINOv2 (ViT-Small)
Instead of starting from scratch (which requires millions of images), we used Meta's **DINOv2** (Vision Transformer). 
*   **Why?** DINOv2 understands texture and depth implicitly. It knows what "vegetation" looks like even if the lighting changes. 
*   **Mechanism**: We extract features from the frozen backbone, creating rich 384-dimensional embeddings for every 14x14 pixel patch of the image.

### 2. The Decoder: Custom U-Net Head
We built a custom lightweight Convolutional Decoder (SeqmentationHead) that takes those transformer features and upsamples them back to the original resolution.
*   **ConvNeXt Blocks**: We use modern convolutional blocks to smooth out the blocky features from the transformer.
*   **Bilinear Upsampling**: Restores the sharp 1080p details needed to spot small rocks or logs.

## 📊 Performance & Evaluation

We evaluated the model on **1002 Unseen Test Images**. This wasn't just a "it looks good" check—we ran strict pixel-level metrics.

### Key Metrics
*   **Mean Pixel Accuracy**: **80.44%** 🎯
*   **Inference Speed**: ~20ms (Real-time capable)
*   **Peak Accuracy**: >92% on clear terrain

### Training Dynamics (Model Learning)
We tracked the model's performance over 5 epochs. The DINOv2 backbone enabled rapid convergence.
| Loss Curve | IoU Curve |
|:---:|:---:|
| ![Loss](Visual_Report/training_curves.png) | ![IoU](Visual_Report/iou_curves.png) |
| *Validation loss stabilized at 0.50* | *mIoU steadily improved to 0.72* |

### quantitative Analysis
Most models fail when the terrain gets messy. Ours maintains high confidence even in complex scenes.

| Accuracy Distribution | Class-Wise Performance |
|:---:|:---:|
| ![Histogram](Visual_Report/accuracy_histogram.png) | ![Class IoU](Visual_Report/class_accuracy_chart.png) |
| *Most frames hit 75-85% accuracy* | *Strong detection of Sky, Landscape, and Background* |

### Qualitative Results (What the Robot Sees)
Below is a direct comparison from our test set. You can see the model (3rd column) successfully identifying the **Trees (Green)** and **Sky (Blue)**, filtering out the noise.

![Performance Summary](Visual_Report/performance_bar_chart.png)

## 🛠️ Installation & Usage

Want to run this on your own machine? It's plug-and-play.

### Prerequisites
*   Python 3.10+
*   CUDA-enabled GPU (Recommended)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Live Demo
We built a GUI tool to let you test individual images and see the metrics in real-time.
```bash
python demo.py
```
*   Press **'O'** to open a file dialog.
*   Select an image from `Offroad_Segmentation_testImages`.
*   Watch the model predict!

### 3. Reproducing Scientific Results
To verify our reported **80.44% Accuracy**, run the evaluation script on the full test set:
```bash
python evaluate_test_set.py
```
This will generate `Test_Optimization_Report.txt` with per-image metrics.

### 4. Interpreting the Visuals
The semantic mask uses specific colors to denote terrain types:
*   🟢 **Green**: Trees & Forest
*   🔵 **Blue**: Sky
*   🟤 **Brown**: Logs & Trunks
*   ⚫ **Black**: Background/Unknown
*   🪨 **Gray**: Rocks
*   🌫️ **Dark Slate**: Distant Landscape


## 📂 Project Structure
*   `src/model.py`: The DINOv2 + U-Net architecture definition.
*   `src/train.py`: Training loop with validation and checkpointing.
*   `src/dataset.py`: Custom PyTorch dataloader for our Off-Road dataset.
*   `evaluate_test_set.py`: Script used to generate the 80% accuracy report.
*   `demo.py`: The presentation-ready GUI application.

---
*Built with ❤️ for the Hackathon.*
