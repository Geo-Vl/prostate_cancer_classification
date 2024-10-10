"""

This script handles the data loading and preprocessing steps for the Prostate Cancer Grade Assessment (PANDA) dataset
or generally histopathology images for training a deep learning model.
It processes image patches, loads labels, applies transformations, and prepares the dataset for training, 
validation, and evaluation of the model. It is designed for histopathology images, categorized 
into 6 classes based on the ISUP grading system (Grades 0-5). The script also includes methods for handling 
class imbalance and provides options for applying transformations.

Key functionalities include:
- Loading image data and corresponding labels from a CSV file.
- Applying transformations such as random augmentations, normalization, padding and tensor conversion.
- Splitting the dataset into training, validation, and evaluation subsets. DataLoader creation.
- Managing class imbalance by computing class weights (alphas) based on the sample distribution.
- Custom collate function for batching images.

Dataset Information:
- The dataset is structured to classify cancer grades (ISUP Grade 0-5), totaling 6 classes.
- Class imbalance is handled by computing class-specific weights based on the number of samples per class.
- Mean and standard deviation values for the RGB channels of the PANDA dataset have been precomputed and are:
  - Mean: [0.803921, 0.596078, 0.729411]
  - Standard Deviation: [0.145098, 0.219607, 0.149019]
- These values can be customized by the user if preferred.

Dependencies:
- torch (PyTorch) for handling tensors and applying transformations.
- torchvision for image transformations.
- pandas for managing the dataset and CSV file operations.
- numpy for numerical operations.
- sklearn (scikit-learn) for data splitting.
- PIL (Python Imaging Library) for opening and handling image files.
- fastai for additional deep learning utilities (optional).


How to Use:
1. Ensure the paths to the dataset (images) and labels (CSV file) are correctly specified.
2. Call the `extract_zip()` function to extract image patches if not already extracted (from `utils.py`).
3. Load and preprocess the data using the `get_x()`, `get_y()`, `open_images()` and `custom_splitter()` functions.
4. Customize the normalization parameters if different dataset statistics are preferred and modify constants like patch_size, batch_size according to your setup.
5. Integrate with your model training script to feed the processed data into your neural network.
"""


import os
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter

# Constants
patch_size = 256  # Size of image patches
batch_size = 10  # Batch size for DataLoader
n_patches = 14  # Number of patches per image
TRAIN = 'path_to_training_directory'  # Path to training image directory
LABELS = 'path_to_labels_csv_file.csv'  # Path to CSV file with image labels

# Normalization values (mean and std) for the dataset, calculated based on image statistics
mean = torch.tensor([0.803921, 0.596078, 0.729411])  # Mean pixel values for RGB channels
std = torch.tensor([0.145098, 0.219607, 0.149019])   # Standard deviation for RGB channels

# Class distribution used for class weighting
class_counts = [2893, 2666, 1344, 1243, 1250, 1225]
total = sum(class_counts)

# Compute the alphas for class weighting
alphas = [total / count for count in class_counts]
alphas_sum = sum(alphas)
normalized_alphas = [alpha / alphas_sum for alpha in alphas]

# Scaling the alphas for balanced class distribution
min_alpha = min(normalized_alphas)
scaled_alphas = [alpha / min_alpha for alpha in normalized_alphas]

# Data Preparation: Reading the dataset CSV file
df = pd.read_csv(LABELS).set_index('image_id')
files = sorted(set([p[:32] for p in os.listdir(TRAIN)]))  # Extract unique image IDs
df = df.loc[files].reset_index()  # Filter dataset based on available files

# Split dataset into train, validation, and evaluation sets
rest_df, eval_df = train_test_split(df, test_size=0.1, stratify=df['isup_grade'], random_state=42)
rest_df = rest_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)

# Further split rest_df into training and validation sets
train_df, valid_df = train_test_split(rest_df, test_size=0.15, stratify=rest_df['isup_grade'], random_state=42)
train_df['split'] = 0  # Mark as training
valid_df['split'] = 1  # Mark as validation
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# Combine train and validation datasets
df_final = pd.concat([train_df, valid_df]).reset_index(drop=True)

# Utility Functions

def get_x(r):
    """Fetches the image file paths for each patch of a given image."""
    return [Path(TRAIN) / f'{r["image_id"]}_{i}.png' for i in range(n_patches)]

def get_y(r):
    """Fetches the label (ISUP grade) for the given image."""
    return r['isup_grade']

def open_images_eval(fn):
    """
    Opens and preprocesses evaluation images with padding and normalization.
    
    Args:
    - fn: List of image file paths.

    Returns:
    - List of processed image tensors.
    """
    processed_imgs = []
    for f in fn:
        img = Image.open(f).convert('RGB')
        padding = (0, 0, max(0, patch_size - img.width), max(0, patch_size - img.height))
        tfms = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.Pad(padding, fill=(255, 255, 255), padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        img_tensor = tfms(img)
        processed_imgs.append(img_tensor)
    return processed_imgs

def open_images(fn):
    """
    Opens and preprocesses training/validation images with augmentations (flips, padding) 
    and normalization.
    
    Args:
    - fn: List of image file paths.

    Returns:
    - List of processed image tensors.
    """
    processed_imgs = []
    for f in fn:
        img = Image.open(f).convert('RGB')
        tfms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(patch_size),
            transforms.Pad((0, 0, max(0, patch_size - img.width), max(0, patch_size - img.height)), 
                           fill=(255, 255, 255), padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        img_tensor = tfms(img)
        processed_imgs.append(img_tensor)
    return processed_imgs

def collate(batch):
    """Custom collate function to stack the image tensors for batching."""
    batch = [item for sublist in batch for item in sublist]  # Flatten the list of lists
    batch = torch.stack(batch)
    return batch

def custom_splitter():
    """
    Custom splitter for training and validation data.
    
    Returns:
    - Two lists: indices for training and validation datasets.
    """
    def _inner(_):
        train_idxs = df_final.index[df_final['split'] == 0].tolist()
        valid_idxs = df_final.index[df_final['split'] == 1].tolist()
        return train_idxs, valid_idxs
    return _inner

# DataBlock Definition and DataLoader Creation

# Training and Validation DataBlock
dblock = DataBlock(
    blocks=(TransformBlock(type_tfms=open_images), CategoryBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=custom_splitter(),
    batch_tfms=[]
)

# Evaluation DataBlock (No augmentations)
dblock_eval = DataBlock(
    blocks=(TransformBlock(type_tfms=open_images_eval), CategoryBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=RandomSplitter(valid_pct=0)
)

"""
# The DataLoaders (dls and dls_eval) can be used directly in the training or evaluation loops as follows.

dls = dblock.dataloaders(df_final, bs=batch_size, collate_fn=collate)
dls_eval = dblock_eval.dataloaders(eval_df, bs=batch_size, collate_fn=collate)
"""
