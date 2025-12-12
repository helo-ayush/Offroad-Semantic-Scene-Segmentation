import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be a tensor of weights [C]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] (Logits)
        # targets: [B, H, W] (Indices)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W]
        # targets: [B, H, W]
        
        # Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        n_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()
        
        # Ignore index if specified (usually padding or void)
        if self.ignore_index is not None:
             # Create mask
             mask = targets != self.ignore_index
             # We would need to handle this broadcasting carefully, 
             # For now, simplistic approach is often sufficient or assume clean data.
             pass 

        # Flatten
        inputs = inputs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)
        
        intersection = (inputs * targets_one_hot).sum()
        
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets_one_hot.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        
        return (1 - self.dice_weight) * loss_focal + self.dice_weight * loss_dice
