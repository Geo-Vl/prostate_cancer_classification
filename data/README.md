# Data Directory

This directory contains the data required for the project. Due to file size limitations, the dataset is not included in this repository. Please follow the instructions below to download and set up the dataset.

## Dataset Information

- **Dataset Source**: The dataset used in this project is from the [PANDA Challenge on Kaggle](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment).
- **Dataset Contents**: The dataset includes:
  - High-resolution pathology images of prostate tissue (`.tiff` files).
  - Labels corresponding to the severity grade of cancer for each image (`train.csv`).
  - Mask files indicating regions of interest (`.tiff` masks).

## Setting Up the Dataset

1. Download the dataset from the [PANDA Challenge Dataset Page](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data).
   
   OR
   
2. Use the Kaggle API to download the dataset
   
   ### Step 1: Install Kaggle API
    If you haven't already, you can install the Kaggle API by running:
     
    ```bash
    pip install kaggle

