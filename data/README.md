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
      
    pip install kaggle

### Step 2: Set Up Your Kaggle API Credentials

  - Go to your Kaggle account and navigate to **Account** settings.
  - Under the **API section**, click on **Create New API Token**.
  - This will download a file named kaggle.json, which contains your API credentials.

  For Linux and MacOS:
  Move the kaggle.json file to a hidden .kaggle directory in your home folder:

    mkdir -p ~/.kaggle
    mv ~/<downloads-folder>/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    
  For Windows:
  Create a folder named .kaggle in C:\Users\<Your-Username>\. Move the kaggle.json file into the .kaggle folder.

