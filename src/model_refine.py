import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ConvBlock(nn.Module):
    """
    Helper module: Conv -> BN -> ReLU
    Used to refine features at each upsampling step.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ProgressiveSemanticSegmenter(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        
        # 1. Robust Backbone Loading (Local Cache Fallback)
        print("Loading DINOv2 backbone (Refined)...")
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f"Online load failed, trying local cache... Error: {e}")
            hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov2_main")
            if os.path.exists(hub_dir):
                self.backbone = torch.hub.load(hub_dir, 'dinov2_vits14', source='local')
                print(">> Loaded DINOv2 from local cache!")
            else:
                raise RuntimeError("Critical: Could not load DINOv2.") from e
        
        # Freeze backbone to avoid destroying pretrained filters
        # We might unfreeze later, but for small datasets, freezing is safer.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval()
        
        self.embedding_dim = 384 # ViT-Small
        
        # 2. Progressive Decoder Stages
        # We need to bridge 384 channels -> Classes
        # And Spatial 1/14 -> 1/1
        
        # Stage 1: Adapter (384 -> 256)
        self.adapter = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Stage 2: 1/14 -> 1/7 (2x up)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(256, 128)
        
        # Stage 3: 1/7 -> 1/3.5 (2x up)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(128, 64)
        
        # Stage 4: 1/3.5 -> 1/1.75 (2x up)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(64, 32)
        
        # Final Stage: Restore to full resolution & Classify
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x):
        # x: [B, 3, H, W] (e.g., 252, 252)
        B, C, H, W = x.shape
        
        # 1. Feature Extraction
        with torch.no_grad():
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict["x_norm_patchtokens"]
            
            # Reshape: (B, N, 384) -> (B, 384, H/14, W/14)
            h_grid = H // 14
            w_grid = W // 14
            patch_tokens = patch_tokens.reshape(B, h_grid, w_grid, self.embedding_dim)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2) # [B, 384, 18, 18]
            
        # 2. Decoding
        x1 = self.adapter(patch_tokens) # [B, 256, 18, 18]
        
        x2 = self.up1(x1)               # [B, 256, 36, 36]
        x2 = self.dec1(x2)              # [B, 128, 36, 36]
        
        x3 = self.up2(x2)               # [B, 128, 72, 72]
        x3 = self.dec2(x3)              # [B, 64, 72, 72]
        
        x4 = self.up3(x3)               # [B, 64, 144, 144]
        x4 = self.dec3(x4)              # [B, 32, 144, 144]
        
        # Final Upsample to Target Size [B, 32, 252, 252]
        # We use interpolate here to handle any slight non-power-of-2 mismatch perfectly
        x_final = F.interpolate(x4, size=(H, W), mode='bilinear', align_corners=False)
        
        # Classifier
        logits = self.final_conv(x_final) # [B, n_classes, 252, 252]
        
        return logits
