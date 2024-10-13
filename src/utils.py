"""

This script contains various utility functions and classes used throughout the project for
tasks like unzipping data, loading and visualizing images, calculating evaluation metrics,
and processing predictions. Functions are consistent with the split in the pre-processing part of the current project.

Key functionalities:
- Unzipping dataset files
- Random image selection and visualization of patches
- Custom multiclass ROC AUC metric calculation
- Generating predictions, calculating Cohen's Kappa, confusion matrix, and ROC AUC

Dependencies:
- torch
- sklearn
- matplotlib
- numpy
- zipfile
- os

Instructions:
1. Provide the correct paths for data loading and unzipping.
2. Ensure that the pre-processing is consistent with the respective from the current project, otherwise adapt (e.g. dl=dls_eval.train).
3. Each utility is modular and can be integrated independently into the overall pipeline.
"""

import zipfile
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score
import torch.nn.functional as F
from fastai.learner import Metric
from fastai.vision.all import *

# --------------------------------------------------
# Utility Function: Unzip dataset
# --------------------------------------------------
def unzip_data(zip_file_path, target_folder):
    """
    Unzips the provided dataset zip file into the target folder.
    Args:
    - zip_file_path (str): Path to the zip file
    - target_folder (str): Directory where the unzipped files will be extracted
    
    Returns:
    - None
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Unzipping the dataset
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    print(f'Files extracted to {target_folder}')

# Example usage
# unzip_data('/kaggle/input/pnd-14-256x256/train_patches.zip', '/kaggle/working/train')

# --------------------------------------------------
# Utility Function: Select and display random image patches
# --------------------------------------------------
def display_random_patches(df_final, get_x, open_images, mean, std):
    """
    Selects a random image, loads its patches, and displays them in a grid format.
    Args:
    - df_final (DataFrame): DataFrame containing image information
    - get_x (function): Function to get image patch paths from a row of df_final
    - open_images (function): Function to load image patches from file paths
    - mean (Tensor): Mean values for normalization
    - std (Tensor): Standard deviation values for normalization
    
    Returns:
    - None
    """
    # Select a random image
    random_index = random.randint(0, len(df_final) - 1)
    random_image_row = df_final.iloc[random_index]

    # Load the image patches
    image_paths = get_x(random_image_row)  # Get paths for the patches
    image_patches = open_images(image_paths)  # Load and preprocess patches

    # Plot the patches
    fig, axes = plt.subplots(2, 7, figsize=(35, 10))  # Adjust grid size based on patches
    axes = axes.flatten()

    for i, img_tensor in enumerate(image_patches):
        img = img_tensor.numpy().transpose((1, 2, 0))  # Convert tensor to numpy array and rearrange dimensions
        img = img * std.numpy() + mean.numpy()  # Un-normalize using mean and std
        img = np.clip(img, 0, 1)  # Clip values to valid range for display
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
# display_random_patches(df_final, get_x, open_images, mean, std)

# --------------------------------------------------
# Metric: Multiclass ROC AUC Calculation
# --------------------------------------------------
class MulticlassRocAuc(Metric):
    """
    Custom metric class to calculate the multiclass ROC AUC score during training or evaluation.
    Uses the 'one-vs-rest' (ovr) approach for multiclass classification.

    Args:
    - None
    """
    def __init__(self):
        self.preds = []
        self.targets = []

    def reset(self):
        """Reset stored predictions and targets."""
        self.preds = []
        self.targets = []

    def accumulate(self, learn):
        """Accumulate predictions and targets during training."""
        preds, targs = learn.pred, learn.y
        self.preds.append(preds)
        self.targets.append(targs)

    @property
    def value(self):
        """Calculate and return the final ROC AUC score."""
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        # Convert to one-hot format for multiclass ROC AUC calculation
        targets_one_hot = F.one_hot(targets, num_classes=preds.size(-1))
        return roc_auc_score(targets_one_hot.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')

# Example usage:
# roc_auc_multiclass = MulticlassRocAuc()

# --------------------------------------------------
# Evaluation: Generate predictions and calculate metrics
# --------------------------------------------------
def evaluate_model(learn, dls_eval):
    """
    Generates predictions on the evaluation dataset, calculates the Cohen's Kappa score,
    confusion matrix, and multiclass ROC AUC score.
    
    Args:
    - learn: The trained model/learner
    - dls_eval: DataLoader for evaluation
    
    Returns:
    - None (prints evaluation results)
    """
    # Generate predictions
    preds, targs = learn.get_preds(dl=dls_eval.train)
    
    # Convert predictions to class indices
    pred_classes = torch.argmax(preds, dim=1)
    
    # Calculate Cohen's Kappa Score
    kappa_score = cohen_kappa_score(targs.numpy(), pred_classes.numpy(), weights='quadratic')
    
    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(targs.numpy(), pred_classes.numpy())
    
    # Convert softmax probabilities to class probabilities for each class
    preds_probs = F.softmax(preds, dim=1)
    
    # Convert targets to one-hot encoding to match the shape of preds_probs for ROC AUC calculation
    targs_one_hot = F.one_hot(targs, num_classes=preds_probs.size(-1))
    
    # Calculate ROC AUC score for multiclass classification
    roc_auc = roc_auc_score(targs_one_hot.cpu().numpy(), preds_probs.cpu().numpy(), multi_class='ovr')
    
    # Output the results
    print(f"ROC AUC Score (Multiclass): {roc_auc}")
    print(f"Kappa Score: {kappa_score}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Example usage:
# evaluate_model(learn, dls_eval)
