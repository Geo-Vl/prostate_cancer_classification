"""

This script defines a deep learning model architecture, custom activation function, patch aggregation 
mechanism, and a custom loss function tailored for patch-based histopathological image analysis. The model is based 
on a modified DenseNet with Mish activation and patch aggregation. A Focal Loss function is 
implemented for handling class imbalances.

Key functionalities:
- Patch-based image aggregation mechanism for better feature extraction.
- DenseNet-based encoder with pooling and final classification layers.
- Custom Focal Loss function with class weighting for handling imbalanced data.

Dependencies:
- torch
- torchvision
- torch.nn
- torch.autograd
- torch.functional as F

Instructions:
1. Ensure all dependencies are installed.
2. Customize the gamma and reduction in Focal Loss if necessary.
3. Import this script as a module in your training pipeline or test it independently.

"""

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Function
from torchvision.models import densenet169, DenseNet169_Weights


# Custom Mish Activation Function
class MishFunction(torch.autograd.Function):
    """
    Mish activation function for enhanced gradient flow.
    The Mish function is defined as: x * tanh(softplus(x))
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    """
    A module wrapper for the Mish activation function, making it compatible with nn.Module layers.
    """
    def forward(self, x):
        return MishFunction.apply(x)


# Helper function to replace ReLU with Mish in any given model
def to_Mish(model):
    """
    Recursively replaces all ReLU activations in the model with the Mish activation function.
    
    Args:
    - model: The input neural network model where ReLU activations will be replaced.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


# Patch Aggregation Class
class PatchAggregator(nn.Module):
    """
    Aggregates features from multiple image patches into a unified map. 
    Uses a learnable weighted sum of the patches for aggregation.
    
    Args:
    - in_features: Number of input features from the encoder.
    """
    def __init__(self, in_features):
        super().__init__()
        self.aggregation = nn.Sequential(
            nn.Linear(in_features, 128),
            Mish(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
        - x: A tensor of patch features (batch_size, n_patches, feature_dim).
        
        Returns:
        - Aggregated tensor (batch_size, feature_dim).
        """
        weights = self.aggregation(x)  # Learnable weights for each patch
        return (x * weights).sum(dim=1)  # Weighted sum of patches


# Pooling Layer (combining Adaptive Average and Max Pooling)
class Pooling(nn.Module):
    """
    Custom pooling layer that concatenates adaptive average and max pooling results.
    """
    def __init__(self):
        super().__init__()
        self.output_size = (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        """
        Applies both max pooling and average pooling to the input tensor, 
        then concatenates the results.
        
        Args:
        - x: Input tensor (batch_size, channels, height, width).
        
        Returns:
        - Pooled tensor (batch_size, 2*channels, 1, 1).
        """
        return torch.cat([self.mp(x), self.ap(x)], 1)


# DenseNet-based Model Architecture
class DenseNetModel(nn.Module):
    """
    A modified DenseNet model with patch-based aggregation, pooling, and custom head for classification.
    
    Args:
    - n_classes: Number of output classes for classification.
    """
    def __init__(self, n_classes=6):
        super().__init__()
        # Load pretrained DenseNet169 model and strip the final classification layer
        weights = DenseNet169_Weights.DEFAULT
        base_model = densenet169(weights=weights)
        layers = list(base_model.children())[:-1]  # Remove the classification head
        feature_dim = list(base_model.children())[-1].in_features

        # Define the encoder using the DenseNet layers
        self.encoder = nn.Sequential(*layers)

        # Custom pooling and aggregation mechanisms
        self.pooling = Pooling()
        self.aggregation = PatchAggregator(2 * feature_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, input):
        """
        Forward pass through the network. Processes a list of image patches, encodes them using DenseNet, 
        pools the features, aggregates them, and then classifies.
        
        Args:
        - input: List of image patches.
        
        Returns:
        - Classification scores for each image (batch_size, n_classes).
        """
        n_patches = len(input)
        batch_size, c, h, w = input[0].size()

        # Stack patches and pass through the encoder
        stacked_patches = torch.stack(input, 1).view(-1, c, h, w)
        patch_features = self.encoder(stacked_patches)

        # Apply pooling and aggregation
        reduced_features = self.pooling(patch_features).view(batch_size, n_patches, -1)
        aggregated_features = self.aggregation(reduced_features)

        # Pass through the classification head
        return self.head(aggregated_features)


# Custom Focal Loss for Handling Class Imbalance
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance in multi-class classification.
    
    Args:
    - alphas: Class-specific weighting factors.
    - gamma: Focusing parameter to down-weight easy examples and focus on hard examples.
    - reduction: Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
    """
    def __init__(self, alphas, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alphas = torch.tensor(alphas).float()

    def forward(self, inputs, targets):
        """
        Compute Focal Loss between input predictions and target labels.
        
        Args:
        - inputs: Predictions (logits) from the model.
        - targets: Ground truth class labels.
        
        Returns:
        - Computed Focal Loss value.
        """
        alphas = self.alphas.to(inputs.device)
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha = alphas[targets]
        F_loss = alpha * ((1 - pt) ** self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

"""
# Example of initializing Focal Loss with class weighting

focal_loss = FocalLoss(alphas=scaled_alphas, gamma=2, reduction='mean')
"""
