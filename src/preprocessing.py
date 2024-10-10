"""

This script handles the data loading and preprocessing steps for the Prostate Cancer Grade Assessment (PANDA) dataset.
It processes image patches, loads labels, applies transformations, and prepares the dataset for training, 
validation, and evaluation of the model. It is designed histopathology images, categorized 
into 6 classes based on the ISUP grading system (Grades 0-5). The script also includes methods for handling 
class imbalance and provides options for applying transformations.

Key functionalities include:
- Extracting image patches from a zip archive.
- Loading image data and corresponding labels from a CSV file.
- Applying transformations such as random augmentations, normalization, padding, and tensor conversion.
- Splitting the dataset into training, validation, and evaluation subsets.
- Managing class imbalance by computing class weights (alphas) based on the sample distribution.

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
- cv2 (OpenCV) for additional image manipulations (if needed).
- matplotlib for visualizing data samples (if needed).
- fastai for additional deep learning utilities (optional).

How to Use:
1. Ensure the paths to the dataset and labels (CSV file) are correctly specified.
2. Call the `extract_zip()` function to extract image patches if not already extracted.
3. Load and preprocess the data using the `load_data()`, `split_data()`, `get_x()`, `get_y()`, and `open_images()` functions.
4. Customize the normalization parameters if different dataset statistics are preferred.
5. Integrate with your model training script to feed the processed data into your neural network.
"""


import os
import zipfile
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
PATCH_SIZE = 256  # Size of image patches
N_PATCHES = 14    # Number of patches per image
TRAIN_DIR = 'path_to_training_images_folder'  # Path to the folder containing training images
LABELS_PATH = 'path_to_labels_csv_file.csv'  # Path to the CSV file with image labels

# Normalization constants
MEAN = torch.tensor([0.803921, 0.596078, 0.729411])
STD = torch.tensor([0.145098, 0.219607, 0.149019])

# Load class counts and compute balanced alphas for the focal loss
CLASS_COUNTS = [2893, 2666, 1344, 1243, 1250, 1225]
TOTAL_SAMPLES = sum(CLASS_COUNTS)
ALPHAS = [TOTAL_SAMPLES / count for count in CLASS_COUNTS]

# Normalize alphas
ALPHAS_SUM = sum(ALPHAS)
NORMALIZED_ALPHAS = [alpha / ALPHAS_SUM for alpha in ALPHAS]
MIN_N_ALPHA = min(NORMALIZED_ALPHAS)
SCALED_ALPHAS =  [alpha / MIN_N_ALPHA for alpha in NORMALIZED_ALPHAS]

# Function to extract the zip file containing image patches
def extract_zip(zip_file_path, target_folder):
    """
    Extracts the zip file containing image patches into the target directory.

    Parameters:
        zip_file_path (str): Path to the zip file.
        target_folder (str): Directory where the files will be extracted.
    
    Returns:
        None
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    print(f'Files extracted to {target_folder}')

# Function to load image paths and labels
def load_data():
    """
    Loads image paths and labels from the provided CSV file and prepares a DataFrame for train/test splitting.

    Returns:
        DataFrame: A DataFrame containing image paths and corresponding labels.
    """
    df = pd.read_csv(LABELS_PATH).set_index('image_id')
    image_files = sorted(set([p[:32] for p in os.listdir(TRAIN_DIR)]))
    df = df.loc[image_files].reset_index()
    
    return df

# Function to split the dataset into training, validation, and evaluation sets
def split_data(df, eval_size=0.1, valid_size=0.15, random_state=42):
    """
    Splits the dataset into training, validation, and evaluation sets based on stratified sampling.

    Parameters:
        df (DataFrame): The complete dataset with image paths and labels.
        eval_size (float): The fraction of data reserved for evaluation.
        valid_size (float): The fraction of remaining training data reserved for validation.
        random_state (int): Seed for reproducibility.
    
    Returns:
        Tuple of DataFrames: Training, validation, and evaluation DataFrames.
    """
    rest_df, eval_df = train_test_split(df, test_size=eval_size, stratify=df['isup_grade'], random_state=random_state)
    train_df, valid_df = train_test_split(rest_df, test_size=valid_size, stratify=rest_df['isup_grade'], random_state=random_state)
    
    # Marking splits
    train_df['split'] = 0
    valid_df['split'] = 1

    df_final = pd.concat([train_df, valid_df]).reset_index(drop=True)
    
    return df_final, eval_df

# Function to get image paths for a specific row in the DataFrame
def get_x(row):
    """
    Retrieves the file paths for the image patches corresponding to a given row.

    Parameters:
        row (Series): A row from the DataFrame with image_id.
    
    Returns:
        List[Path]: A list of file paths for the image patches.
    """
    return [Path(TRAIN_DIR) / f'{row["image_id"]}_{i}.png' for i in range(N_PATCHES)]

# Function to get labels (ISUP grades) for a specific row in the DataFrame
def get_y(row):
    """
    Retrieves the label (ISUP grade) corresponding to a given row.

    Parameters:
        row (Series): A row from the DataFrame with the isup_grade column.
    
    Returns:
        int: The ISUP grade for the given row.
    """
    return row['isup_grade']

# Function to open and preprocess image patches (with augmentations for training)
def open_images(file_paths):
    """
    Opens and processes a list of image file paths with random augmentations and normalization.

    Parameters:
        file_paths (List[str]): List of image file paths.

    Returns:
        List[Tensor]: A list of processed image tensors.
    """
    processed_imgs = []
    
    for f in file_paths:
        img = Image.open(f).convert('RGB')
        
        tfms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(PATCH_SIZE),
            transforms.Pad((0, 0, max(0, PATCH_SIZE - img.width), max(0, PATCH_SIZE - img.height)), fill=(255, 255, 255), padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        
        img_tensor = tfms(img)
        processed_imgs.append(img_tensor)

    return processed_imgs

# Function to handle evaluation data loading without augmentation
def open_images_eval(file_paths):
    """
    Opens and processes a list of image file paths for evaluation (without augmentations).

    Parameters:
        file_paths (List[str]): List of image file paths.

    Returns:
        List[Tensor]: A list of processed image tensors.
    """
    processed_imgs = []
    
    for f in file_paths:
        img = Image.open(f).convert('RGB')
        
        tfms = transforms.Compose([
            transforms.CenterCrop(PATCH_SIZE),
            transforms.Pad((0, 0, max(0, PATCH_SIZE - img.width), max(0, PATCH_SIZE - img.height)), fill=(255, 255, 255), padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        
        img_tensor = tfms(img)
        processed_imgs.append(img_tensor)

    return processed_imgs

# Custom collate function to handle batched data
def collate(batch):
    """
    Collates a batch of image patches into a single batch tensor.

    Parameters:
        batch (List): A list of lists, each containing image patches.

    Returns:
        Tensor: A batched tensor of image patches.
    """
    batch = [item for sublist in batch for item in sublist]  # Flatten the list of lists
    return torch.stack(batch)
