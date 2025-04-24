import torch
import torch.nn.functional as F

def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    return 1 - dice

def binary_accuracy(preds, targets):
    preds = torch.sigmoid(preds)
    preds_bin = (preds > 0.5).float()
    correct = (preds_bin == targets).float().sum()
    total = torch.numel(targets)
    return correct / total

def compute_dice(pred, target, eps=1e-8):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def compute_accuracy(pred, target):
    pred = (pred > 0.5).float()
    return (pred == target).float().mean()
