"""

This script handles the training process for a DenseNet-based model using the FastAI library.
It includes the definition of the learner, usage of custom loss functions, optimizers, and metrics,
as well as callbacks for early stopping (based on valid loss values) and model saving (based on cohen kappa score).

Key functionalities:
- Model training using a custom DenseNet architecture with Mish activation function
- Focal loss implementation for handling class imbalance
- Ranger optimizer for training stability
- Gradient clipping for preventing exploding gradients
- Learning rate finder for identifying the optimal learning rate
- Early stopping and best model saving during training

Dependencies:
- fastai
- torch

Instructions:
1. Modify the data paths and any custom settings as per your data.
2. Run the script to train the model, using the pre-defined learner and callbacks.
"""

from fastai.vision.all import *
from model import DenseNetModel  # Import custom model from model.py
from utils import MulticlassRocAuc  # Import custom metrics from utils.py

# --------------------------------------------------
# Model Training Definition
# --------------------------------------------------

def train_model(dls, focal_loss, lr=None, epochs, patience, lr_find=False):
    """
    Trains the DenseNet-based model using FastAI's Learner.

    Args:
    - dls (DataLoaders): The DataLoaders object that contains the training and validation data.
    - focal_loss (Loss function): The custom Focal Loss function to handle class imbalance.
    - lr (float, optional): The learning rate for training. If None, the lr_find() method will be used.
    - epochs (int): Number of epochs to train the model for.
    - patience (int): The patience for early stopping.
    - lr_find (bool, optional): If True, the learning rate finder will be executed. Defaults to False.

    Returns:
    - learner (Learner): The trained FastAI Learner object.
    """
    # Initialize the model
    model = DenseNetModel()

    # Define custom metrics
    roc_auc_multiclass = MulticlassRocAuc()  # Custom ROC AUC metric for multiclass classification
    kappa_metric = CohenKappa(weights='quadratic')  # Cohen's Kappa with quadratic weights

    # Define the optimizer and learner
    learn = Learner(dls, model, loss_func=focal_loss, opt_func=Ranger, 
                    metrics=[roc_auc_multiclass, kappa_metric])

    # Enable mixed precision (float16) training for better performance
    learn.to_fp16()

    # Gradient clipping to prevent exploding gradients
    learn.clip_grad = 1.0

    # Find optimal learning rate if requested
    if lr_find:
        learn.lr_find()
        # Note: After calling lr_find(), you should manually set the learning rate based on the plot.
        return None

    # If no learning rate is provided, set a default value
    if lr is None:
        raise ValueError("Learning rate not specified. Run with lr_find=True to identify an optimal learning rate.")

    # Callbacks for early stopping and saving the best model
    early_stop_cb = EarlyStoppingCallback(monitor='valid_loss', patience=patience)
    save_model_cb = SaveModelCallback(monitor='cohen_kappa_score', fname='best_model', with_opt=True)

    # Train the model using the 1-cycle policy
    learn.fit_one_cycle(epochs, lr, cbs=[early_stop_cb, save_model_cb])

    # Load the best saved model based on Cohen's Kappa score
    learn.load('best_model')

    return learn

"""
# --------------------------------------------------
# Example Usage
# --------------------------------------------------
if __name__ == "__main__":
    # Assume `dls` is the DataLoaders object containing the training/validation data.
    # Also assume `focal_loss` has been defined as per model.py.
    
    # You would need to define or load your DataLoaders (`dls`) here. 
    # Ensure that your dataset is preprocessed correctly.
    
    try:
        # Train the model
        learner = train_model(
            dls=dls,                 # DataLoaders object
            focal_loss=focal_loss,    # Focal loss (from model.py)
            lr=0.0003,                # The learning rate identified via lr_find()
            epochs=30,                # Number of epochs to train
            patience=24,              # Early stopping patience
            lr_find=False             # Set to True if you want to perform learning rate search
        )
"""
