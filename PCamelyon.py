#!/usr/bin/env python
# coding: utf-8

# Imports
import yaml
import logging
import argparse
import os
import sys
import torchvision
import torch
import datasets
from datasets import load_dataset
import transformers
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import numpy as np
import collections # For Counter
from tqdm import tqdm # For progress bars

import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import torch.optim as optim
import warnings

# --- Global Settings / Constants (can be at top level or moved into main/config later) ---
# BATCH_SIZE = 64 # Can be made an arg or config later
# DEFAULT_LEARNING_RATE = 1e-4

# --- Helper Functions (remain at top level for importability) ---

# Custom Plotting Function
def display_or_save_plot(figure, filename_base, hpc_mode_flag, output_path_base):
    """
    Displays a Matplotlib plot or saves it to a file based on hpc_mode.

    Args:
        figure (matplotlib.figure.Figure): The Matplotlib figure object to display/save.
                                           If None, uses plt.gcf() (get current figure).
        filename_base (str): The base name for the saved file (e.g., "learning_curves").
                             Will have ".png" appended.
        hpc_mode_flag (bool): If True, saves the plot. If False, shows the plot.
        output_path_base (str): The base directory where plots will be saved.
    """
    if figure is None:
        figure = plt.gcf()
    if hpc_mode_flag:
        plot_dir = os.path.join(output_path_base, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"{filename_base}.png")
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        logging.info(f"Plot saved to: {save_path}")
        plt.close(figure)
    else:
        plt.show()

def count_labels(dataset):
    '''Count the number of samples for each label in the dataset.'''
    label_counts = {0: 0, 1: 0}
    for sample in dataset:
        label_counts[sample['label']] += 1
    return label_counts

def show_samples(dataset, num_samples_per_class=5, hpc_mode_flag=False, output_path_base="."):
    """
    Efficiently displays a few random sample images from the dataset for each class.
    Args:
        dataset (datasets.Dataset): A HuggingFace dataset split (e.g., pcam['train']).
        num_samples_per_class (int): Number of samples to show for each class.
        hpc_mode_flag (bool): If True, saves the plot. If False, shows the plot.
        output_path_base (str): The base directory where plots will be saved.
    """
    # Attempt to get class names from features for nice titles
    try:
        class_feature = dataset.features['label']
        class_names_map = {
            False: class_feature.names[0] if hasattr(class_feature, 'names') else "Class False/0 (No Tumor)",
            True: class_feature.names[1] if hasattr(class_feature, 'names') else "Class True/1 (Tumor)"
        }
        target_labels_for_display = [False, True] # We want to display 'no_tumor' then 'tumor'
    except (KeyError, AttributeError):
        logging.warning("Warning: Could not automatically determine class names. Using raw labels.")
        class_names_map = {False: "Label False", True: "Label True"}
        target_labels_for_display = [False, True]

    fig, axes = plt.subplots(len(target_labels_for_display), num_samples_per_class,
                             figsize=(num_samples_per_class * 3, len(target_labels_for_display) * 3))

    # Ensure axes is always 2D for consistent indexing, even if only one class or one sample
    if len(target_labels_for_display) == 1 and num_samples_per_class == 1:
        axes = np.array([[axes]])
    elif len(target_labels_for_display) == 1:
        axes = np.array([axes])
    elif num_samples_per_class == 1:
        axes = axes.reshape(-1,1)


    all_labels = dataset['label']

    for i, label_to_display in enumerate(target_labels_for_display):
        label_specific_indices = [idx for idx, actual_label in enumerate(all_labels) if actual_label == label_to_display]

        if not label_specific_indices:
            logging.info(f"No samples found for label: {class_names_map[label_to_display]}")
            for j in range(num_samples_per_class):
                if i < axes.shape[0] and j < axes.shape[1]: # Check bounds
                    ax = axes[i, j]
                    ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
                    ax.axis('off')
            continue

        # Shuffle indices to get random samples
        np.random.shuffle(label_specific_indices)

        # Select up to num_samples_per_class
        indices_to_show = label_specific_indices[:num_samples_per_class]

        for j, sample_idx in enumerate(indices_to_show):
            sample = dataset[sample_idx] # Fetch the actual sample using the pre-identified index
            img = sample['image'] # PIL image

            if i < axes.shape[0] and j < axes.shape[1]: # Check bounds
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_title(f"{class_names_map[label_to_display]}", fontsize=10)
                ax.axis('off')
            else: # Should not happen with correct subplot setup but good for safety
                logging.warning(f"Warning: Axes index out of bounds for plot [{i},{j}]")

    plt.suptitle("Sample Images from Dataset", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    display_or_save_plot(plt.gcf(), "sample_images", hpc_mode_flag, output_path_base)

class PatchCamelyonDataset(Dataset):
    """ Custom Dataset class for PatchCamelyon dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = int(item['label']) # Convert boolean to int
        if self.transform:
            image = self.transform(image)
        # For BCEWithLogitsLoss, if model output is [batch_size, 1], target should be [batch_size, 1]
        label = torch.tensor([label], dtype=torch.float32) # Make it [1] shape
        return image, label

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False, disable=HPC_MODE) # Optionally disable tqdm in HPC_MODE if it clutters logs

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs) # Shape: [batch_size, 1] (logits)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)

        # For accuracy: convert logits to probabilities, then to predictions
        probs = torch.sigmoid(outputs) # Shape: [batch_size, 1]
        preds = (probs > 0.5).float()  # Shape: [batch_size, 1], converts to 0.0 or 1.0

        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if not HPC_MODE:  # Only update tqdm postfix if not in HPC to avoid too much log clutter
            progress_bar.set_postfix(loss=loss.item(),
                                     acc=correct_predictions / total_samples if total_samples > 0 else 0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch.
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False, disable=HPC_MODE)

    with torch.no_grad():  # No need to track gradients during validation
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if not HPC_MODE:
                progress_bar.set_postfix(loss=loss.item(),
                                         acc=correct_predictions / total_samples if total_samples > 0 else 0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def plot_learning_curves(history, title_suffix="", hpc_mode_flag=False, output_path_base="."):
    """
    Plots the learning curves for training and validation loss and accuracy.
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        title_suffix (str): Suffix to add to the plot title.
        hpc_mode_flag (bool): If True, saves the plot instead of showing it.
        output_path_base (str): Base directory where plots will be saved.
    """
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Get figure object

    ax1.plot(epochs_range, history['train_loss'], label='Training Loss')
    ax1.plot(epochs_range, history['val_loss'], label='Validation Loss')
    ax1.set_title(title_suffix + 'Training and Validation Loss')
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss');
    ax1.legend();
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    ax2.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    ax2.set_title(title_suffix + 'Training and Validation Accuracy')
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('Accuracy');
    ax2.legend();
    ax2.grid(True)

    fig.suptitle('Learning Curves' + (f" - {title_suffix.strip()}" if title_suffix else ""), fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    filename = f"learning_curves{title_suffix.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
    display_or_save_plot(fig, filename.strip('_'), hpc_mode_flag, output_path_base)

def run_lr_experiment(learning_rate, num_epochs, model_architecture_fn,
                      train_loader, val_loader, criterion_class, optimizer_class, device,
                      experiment_name="", hpc_mode_flag=False, output_path_base="."):
    """
    Runs a training experiment for a given learning rate.

    Args:
        learning_rate (float): The learning rate to use.
        num_epochs (int): Number of epochs to train.
        model_architecture_fn (callable): A function that returns an uninitialized model (e.g., models.resnet18).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion_class (callable): The loss function class (e.g., nn.BCEWithLogitsLoss).
        optimizer_class (callable): The optimizer class (e.g., optim.Adam).
        device (torch.device): The device to train on.
        experiment_name (str): A name for this experiment (e.g., "lr_1e-3") for saving files.
        hpc_mode_flag (bool): If True, saves plots instead of showing them.
        output_path_base (str): Base directory where plots/models will be saved.

    Returns:
        tuple: (history, best_validation_accuracy, best_epoch_number)
    """
    logging.info(f"\n--- Running Experiment: {experiment_name} (LR: {learning_rate}) ---")

    # Re-initialize Model
    model = model_architecture_fn()    # TODO: generalize for other architectures (e.g. pass `weights="IMAGENET1K_V1"` as an arg/config -> string enum value)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = criterion_class()
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    best_epoch = 0
    model_dir = os.path.join(output_path_base, "models")
    os.makedirs(model_dir, exist_ok=True)
    path_best_model = os.path.join(model_dir, f"pcam_resnet18_best_model_{experiment_name}.pth")

    logging.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        current_epoch_num = epoch + 1

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logging.info(f"Epoch {current_epoch_num}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = current_epoch_num
            torch.save(model.state_dict(), path_best_model)
            logging.info(f"  🎉 New best model saved for {experiment_name}! Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")

    logging.info(f"\nFinished Training for LR={learning_rate}.")
    logging.info(f"Best val acc for LR {learning_rate}: {best_val_acc:.4f} at Epoch {best_epoch}")
    logging.info(f"Best model saved to: {path_best_model}")

    plot_learning_curves(history, title_suffix=f" (LR={learning_rate}) ")

    return history, best_val_acc, best_epoch


# --- Main function ---
def main(cli_args):
    # Load config
    config = {}
    if cli_args.config:
        try:
            with open(cli_args.config, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from: {cli_args.config}") # Change to logging later
        except FileNotFoundError:
            logging.warning(f"WARNING: Config file {cli_args.config} not found. Using defaults and CLI args.") # Change to logging
        except Exception as e:
            logging.error(f"ERROR: Could not load/parse config file {cli_args.config}: {e}. Exiting.") # Change to logging
            return

    # Config precedence: CLI args override config settings
    final_config = config.copy()

    # Override with CLI arguments if they were provided AND are different from their argparse defaults
    if cli_args.hpc_mode is not None and cli_args.hpc_mode != parser.get_default('hpc_mode'):
        final_config['hpc_mode'] = cli_args.hpc_mode
    else:  # If not set by CLI or same as default, take from config or set a script default
        final_config.setdefault('hpc_mode', 0)

    if cli_args.output_dir is not None and cli_args.output_dir != parser.get_default('output_dir'):
        final_config['output_dir_base'] = cli_args.output_dir
    else:
        final_config.setdefault('output_dir_base', './experiment_outputs_default')

    # Ensure essential parameters have script-level defaults if not in config or CLI
    final_config.setdefault('device_preference', "cuda")
    final_config.setdefault('random_seed', None)  # Or 42
    final_config.setdefault('batch_size', 64)
    final_config.setdefault('num_workers', 0)  # Default to 0 if not specified
    final_config.setdefault('model_architecture', "resnet18")
    final_config.setdefault('baseline_learning_rate', 0.0001)
    final_config.setdefault('baseline_num_epochs', 5)
    final_config.setdefault('optimizer_type', "Adam")
    final_config.setdefault('lr_tuning_num_epochs', 10)
    final_config.setdefault('lrs_to_test', [0.001, 0.00001])

    # Unique output directory based on timestamp
    current_output_dir = final_config['output_dir_base']
    os.makedirs(current_output_dir, exist_ok=True)

    # Logging setup
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    log_file_path = os.path.join(current_output_dir, "run.log")

    # Clear existing handlers if any (important if main is called multiple times in a session)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("************************************************************")
    logging.info("           Starting PatchCamelyon Training Script           ")
    logging.info("************************************************************")
    logging.info(f"Running with final configuration: {final_config}")
    logging.info(f"Output directory for plots/models/logs: {current_output_dir}")
    logging.info(f"Log file: {log_file_path}")

    # Save the final config to a YAML file for reference
    config_save_path = os.path.join(current_output_dir, "effective_config.yaml")
    try:
        with open(config_save_path, 'w') as f:
            yaml.dump(final_config, f, indent=4, sort_keys=False)
        logging.info(f"Effective configuration saved to: {config_save_path}")
    except Exception as e:
        logging.error(f"Could not save effective configuration: {e}")

    # Set random seed for reproducibility if specified
    if final_config['random_seed'] is not None:
        seed = final_config['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        logging.info(f"Random seed set to: {seed}")

    global HPC_MODE # continue to use global to access in helper functions, for now
    HPC_MODE = final_config['hpc_mode'] == 1

    # Set device preference based on config or CLI args
    device_preference = final_config['device_preference'].lower()
    if device_preference == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_preference == "cpu":    # explicitly set to CPU
        device = torch.device("cpu")
    else:       # fallback to best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    logging.info(f"Using device: {device}")


    # Initial Setup
    warnings.filterwarnings("ignore", category=UserWarning)
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 6)

    logging.info("\n--- Phase 0 & 1: Data Loading and Preparation ---")
    try:
        pcam = load_dataset("1aurent/PatchCamelyon")
        logging.info("Successfully loaded PatchCamelyon dataset.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        logging.exception("Exception details:")
        return  # Exit if dataset fails to load


    logging.info("Dataset structure:")
    logging.info(pcam)

    # Calculate the total number of samples in the dataset and distribution across splits
    total_samples = sum(len(pcam[split]) for split in pcam)
    logging.info(f"\nTotal samples in the dataset: {total_samples}")

    # Check the available splits in the dataset
    logging.info("\nAvailable splits in the dataset:")
    for split in pcam:
        split_samples = len(pcam[split])
        logging.info(f"  - {split}: {split_samples} samples - {split_samples / total_samples * 100:.2f}%")

    # Check feature names and types
    logging.info("\nFeature names and types:")
    for split in pcam:
        logging.info(f"\n{split} split features:")
        for feature_name, feature_type in pcam[split].features.items():
            logging.info(f"  - {feature_name}: {feature_type}")

    # Check the first few samples in the training set
    logging.info("\nFirst few samples in the training set:")
    for i in range(3):
        logging.info(f"Sample {i}:")
        logging.info(pcam['train'][i])

    # Check image dimensions, color channels, and label distribution.

    # get the first sample from the training set
    sample = pcam['train'][0]

    image = sample['image']
    label = sample['label']

    logging.info(f"\nFirst training sample image type: {type(image)}")
    logging.info(f"Image dimensions: {image.size} (width x height)")
    logging.info(f"Image mode (color channels): {image.mode}")
    logging.info(f"Label: {label}")

    # Class labels are binary (0 = False for no tumor, 1 = True for tumor).

    # Check if the dataset is balanced in terms of class distribution.
    train_labels_raw = pcam['train']['label']
    raw_counts = collections.Counter(train_labels_raw)
    label_counts_for_plot = {0: raw_counts.get(False, 0), 1: raw_counts.get(True, 0)}
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(label_counts_for_plot.keys()), y=list(label_counts_for_plot.values()),
                palette='viridis', hue=list(label_counts_for_plot.keys()), legend=False, ax=ax_dist)  # Pass ax
    ax_dist.set_title('Label Distribution in Training Set')
    ax_dist.set_xlabel('Label');
    ax_dist.set_ylabel('Number of Samples')
    ax_dist.set_xticks([0, 1], ['No Tumor (0)', 'Tumor (1)'])
    for i, count_val in enumerate(label_counts_for_plot.values()):
        ax_dist.text(i, count_val / 2, f"{count_val}", ha='center', color='white', fontweight='bold')
    fig_dist.tight_layout()
    display_or_save_plot(fig_dist, "label_distribution", HPC_MODE, current_output_dir)

    # Visualize some samples
    show_samples(pcam['train'], num_samples_per_class=3, hpc_mode_flag=HPC_MODE, output_path_base=current_output_dir)

    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),   # Complements RandomRotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Add color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    BATCH_SIZE_ARG = final_config['batch_size']
    train_custom_dataset = PatchCamelyonDataset(pcam['train'], transform=train_transform)
    val_custom_dataset = PatchCamelyonDataset(pcam['valid'], transform=val_test_transform)
    test_custom_dataset = PatchCamelyonDataset(pcam['test'], transform=val_test_transform)

    NUM_WORKERS_ARG = final_config['num_workers']
    train_dataloader = DataLoader(train_custom_dataset, batch_size=BATCH_SIZE_ARG, shuffle=True, num_workers=NUM_WORKERS_ARG,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_custom_dataset, batch_size=BATCH_SIZE_ARG, shuffle=False, num_workers=NUM_WORKERS_ARG,
                                pin_memory=True)
    test_dataloader = DataLoader(test_custom_dataset, batch_size=BATCH_SIZE_ARG, shuffle=False, num_workers=NUM_WORKERS_ARG,
                                 pin_memory=True)

    # --- Initial Model Training (Baseline) ---
    logging.info("\n--- Baseline Model Training ---")
    BASELINE_LEARNING_RATE_ARG = final_config['baseline_learning_rate']
    BASELINE_NUM_EPOCHS_ARG = final_config['baseline_num_epochs']
    OPTIMIZER_ARG = final_config['optimizer_type']

    model = models.get_model(final_config['model_architecture'], weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    if final_config['optimizer_type'].lower() == 'adam':        # TODO: make this more generic for other optimizers, could include parameters in config
        optimizer = optim.Adam(model.parameters(), lr=BASELINE_LEARNING_RATE_ARG)   # add more parameters later if needed
    elif final_config['optimizer_type'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=BASELINE_LEARNING_RATE_ARG, momentum=0.9)  # add more parameters later if needed
    else:
        logging.error(f"Unsupported optimizer type: {final_config['optimizer_type']}")
        return      # Exit if unsupported optimizer

    history_baseline = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_accuracy_baseline = 0.0
    best_epoch_baseline = 0

    model_dir = os.path.join(current_output_dir, "models")  # Centralize model saving
    os.makedirs(model_dir, exist_ok=True)
    path_best_baseline_model = os.path.join(model_dir, "pcam_resnet18_baseline_best.pth")

    logging.info(f"Starting baseline training for {BASELINE_NUM_EPOCHS_ARG} epochs...")
    for epoch in range(BASELINE_NUM_EPOCHS_ARG):
        current_epoch_num = epoch + 1
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_dataloader, criterion, device)

        history_baseline['train_loss'].append(train_loss)
        history_baseline['train_acc'].append(train_acc)
        history_baseline['val_loss'].append(val_loss)
        history_baseline['val_acc'].append(val_acc)

        logging.info(
            f"Epoch {current_epoch_num}/{BASELINE_NUM_EPOCHS_ARG} Baseline | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_accuracy_baseline:
            best_val_accuracy_baseline = val_acc
            best_epoch_baseline = current_epoch_num
            torch.save(model.state_dict(), path_best_baseline_model)
            logging.info(
                f"  🎉 New best baseline model saved! Val Acc: {best_val_accuracy_baseline:.4f} at Epoch {best_epoch_baseline}")

    logging.info("\nFinished Baseline Training.")
    logging.info(f"Best baseline val_acc: {best_val_accuracy_baseline:.4f} at Epoch {best_epoch_baseline}")
    plot_learning_curves(history_baseline, title_suffix="Baseline ", hpc_mode_flag=HPC_MODE,
                         output_path_base=current_output_dir)

    # --- Evaluation of the best baseline model on Test Set ---
    logging.info("\n--- Evaluating Best Baseline Model on Test Set ---")
    eval_model = models.resnet18(weights=None)  # Create a new instance for evaluation
    num_ftrs_eval = eval_model.fc.in_features
    eval_model.fc = nn.Linear(num_ftrs_eval, 1)
    try:
        eval_model.load_state_dict(torch.load(path_best_baseline_model, map_location=device))
        logging.info(f"Successfully loaded best baseline model from {path_best_baseline_model}")
    except Exception as e:
        logging.info(f"Error loading best baseline model: {e}. Skipping test set evaluation for baseline.")
        return  # Or handle differently

    eval_model = eval_model.to(device)
    eval_model.eval()

    all_labels_test, all_predictions_test, all_probabilities_test = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Test Set Eval (Baseline)", disable=HPC_MODE):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = eval_model(inputs)
            probs = torch.sigmoid(outputs)
            all_probabilities_test.extend(probs.cpu().numpy())
            all_predictions_test.extend((probs > 0.5).float().cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    all_labels_test = np.array(all_labels_test).flatten()
    all_predictions_test = np.array(all_predictions_test).flatten()
    all_probabilities_test = np.array(all_probabilities_test).flatten()

    acc_test = accuracy_score(all_labels_test, all_predictions_test)
    prec_test = precision_score(all_labels_test, all_predictions_test, zero_division=0)
    rec_test = recall_score(all_labels_test, all_predictions_test, zero_division=0)
    f1_test = f1_score(all_labels_test, all_predictions_test, zero_division=0)
    roc_auc_test = roc_auc_score(all_labels_test, all_probabilities_test)

    logging.info("\n--- Baseline Model Test Set Performance ---")
    logging.info(f"Accuracy:  {acc_test:.4f}")
    logging.info(f"Precision: {prec_test:.4f}");
    logging.info(f"Recall:    {rec_test:.4f}")
    logging.info(f"F1-score:  {f1_test:.4f}");
    logging.info(f"AUC-ROC:   {roc_auc_test:.4f}")

    # Confusion Matrix Plot for Test Set (Baseline)
    cm_test = confusion_matrix(all_labels_test, all_predictions_test)
    # ... (your enhanced CM plotting logic, adapted to use display_or_save_plot)
    fig_cm_test, ax_cm_test = plt.subplots(figsize=(8, 7))
    group_names = ['TN', 'FP', 'FN', 'TP']  # Simplified for brevity here
    group_counts = ["{0:0.0f}".format(value) for value in cm_test.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm_test.flatten() / np.sum(cm_test)]
    annot_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(2, 2)
    sns.heatmap(cm_test, annot=annot_labels, fmt='', cmap='Blues', ax=ax_cm_test,
                xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
    ax_cm_test.set_title('Confusion Matrix - Test Set (Baseline)')
    ax_cm_test.set_ylabel('Actual');
    ax_cm_test.set_xlabel('Predicted')
    display_or_save_plot(fig_cm_test, "confusion_matrix_baseline_test", HPC_MODE, current_output_dir)

    # ROC Curve for Test Set (Baseline)
    fpr_test, tpr_test, _ = roc_curve(all_labels_test, all_probabilities_test)
    fig_roc_test, ax_roc_test = plt.subplots(figsize=(8, 6))
    ax_roc_test.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'ROC (area = {roc_auc_test:.2f})')
    ax_roc_test.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc_test.set_xlim([0, 1]);
    ax_roc_test.set_ylim([0, 1.05])
    ax_roc_test.set_xlabel('FPR');
    ax_roc_test.set_ylabel('TPR')
    ax_roc_test.set_title('ROC Curve - Test Set (Baseline)');
    ax_roc_test.legend(loc="lower right");
    ax_roc_test.grid(True)
    display_or_save_plot(fig_roc_test, "roc_curve_baseline_test", HPC_MODE, current_output_dir)

    # --- Phase 2: LR Experiments ---
    logging.info("\n--- Phase 2: Learning Rate Experiments ---")
    NUM_EPOCHS_LR_TUNING_ARG = final_config['lr_tuning_num_epochs']
    lrs_to_test = final_config['lrs_to_test']
    model_architecture_fn = lambda: models.get_model(final_config['model_architecture'], weights=models.ResNet18_Weights.IMAGENET1K_V1)

    lr_experiment_results = {}
    # Run for original LR too for a clean comparison within this experimental setup
    all_lrs_for_exp = [BASELINE_LEARNING_RATE_ARG] + lrs_to_test

    for lr_val in all_lrs_for_exp:
        exp_name = f"lr_{lr_val:.0e}".replace('-', '_minus_')
        hist, b_val_acc, b_epoch = run_lr_experiment(
            learning_rate=lr_val,
            num_epochs=NUM_EPOCHS_LR_TUNING_ARG,
            model_architecture_fn=model_architecture_fn,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            criterion_class=nn.BCEWithLogitsLoss,
            optimizer_class=optim.Adam,
            device=device,
            experiment_name=exp_name,
            hpc_mode_flag=HPC_MODE,  # Pass HPC_MODE
            output_path_base=current_output_dir
        )
        lr_experiment_results[lr_val] = {'history': hist, 'best_val_acc': b_val_acc, 'best_epoch': b_epoch}

    logging.info("\n--- All LR Experiments Summary ---")
    for lr_val, result in lr_experiment_results.items():
        logging.info(f"LR: {lr_val:.0e} -> Best Val Acc: {result['best_val_acc']:.4f} at Epoch {result['best_epoch']}")

    logging.info("\n--- Script Finished ---")


# --- Entry point for the script ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="PatchCamelyon Training and Evaluation Script")
    parser.add_argument(
        '--config',
        type=str,
        default=None,  # No default config file, script can run with internal defaults or CLI args
        help="Path to YAML configuration file. CLI arguments will override config file settings."
    )
    parser.add_argument(
        '--hpc_mode', type=int, default=None, choices=[0, 1],
        help="Set to 1 for HPC mode (saves plots). Default 0 (shows plots)."
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,  # Changed default
        help="Base directory to save models, plots, and logs."
    )

    # Conditional parsing for Jupyter compatibility
    if 'ipykernel_launcher.py' in sys.argv[0] or 'colab_kernel_launcher.py' in sys.argv[0]:
        logging.info("Running in Jupyter/Colab mode. Using default arguments for main execution.")
        cli_parsed_args = parser.parse_args(args=[])  # Use defaults for main function
    else:
        cli_parsed_args = parser.parse_args()

    # Call the main execution function
    main(cli_parsed_args)
