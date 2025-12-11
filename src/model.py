import torch
import torch.nn as nn

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW=14, tokenH=14):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        return self.classifier(x)

class OffroadModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        
        print("Loading DINOv2 backbone...")
        try:
            # Try online load first (checks for updates)
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f"Online load failed ({e}), trying local cache...")
            # Fallback to local cache if GitHub is down
            hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov2_main")
            if os.path.exists(hub_dir):
                self.backbone = torch.hub.load(hub_dir, 'dinov2_vits14', source='local')
                print(">> Loaded DINOv2 from local cache!")
            else:
                raise RuntimeError("Could not load DINOv2 from online or local cache.") from e

        self.backbone.eval()
        
        self.n_embedding = 384 # embedding dim for vits14
        
        # Input: 252x252 -> 18x18 tokens (14px patch)
        self.head = SegmentationHeadConvNeXt(self.n_embedding, n_classes)

    def forward(self, x):
        # x: [B, 3, H, W] where H, W must be divisible by 14
        B, C, H, W = x.shape
        
        # Get patch tokens from DINOv2
        features_dict = self.backbone.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"] 
        
        # Reshape tokens to spacial map [B, C, h, w]
        h_token = H // 14
        w_token = W // 14
        
        patch_tokens = patch_tokens.reshape(B, h_token, w_token, self.n_embedding)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)
        
        # Pass through segmentation head
        logits = self.head.stem(patch_tokens)
        logits = self.head.block(logits)
        logits = self.head.classifier(logits)
        
        # Upsample to original resolution
        output = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return output

# Alias for compatibility
UNet = OffroadModel 

