# PatchCamelyon Histology Image Classification: A Learning Project

## Overview

This repository documents a machine learning project focused on binary image classification using the **PatchCamelyon (PCam)** dataset. This project serves as a practical learning exercise to understand and implement a complete ML/DL pipeline for histology image analysis.

The main motivation is to build skills and explore techniques that will be transferable to a future, real-world biomedical challenge: **the detection of *Haplosporidium nelsoni* (MSX) disease in high-resolution histology slide images (WSIs) of oysters.**

## Learning Objectives (for this PCam Project)

This "dummy" project aims to achieve the following learning objectives:

*   Understand and implement a complete ML/DL image classification pipeline.
*   Learn effective data loading, preprocessing, and data augmentation for histology image patches.
*   Gain experience with selecting, fine-tuning, and evaluating pre-trained DL models (e.g., ResNet18).
*   Explore techniques for dealing with potentially imbalanced datasets (conceptual).
*   Practice implementing basic model interpretation/explainability techniques (XAI) (planned).
*   Learn to structure code modularly, manage configurations, and document experiments effectively.
*   Simulate working with a binary classification task relevant to "disease presence/absence."

## The Target MSX Oyster Disease Project (Context)

The ultimate goal this project prepares for involves:

*   **Goal:** Implement an ML/DL system to assist experts in identifying MSX presence in oyster histology slices.
*   **Desired Output:** Signal MSX presence, ideally with bounding boxes around affected areas.
*   **Anticipated Dataset Characteristics (MSX):** Large WSI dimensions, H&E-like staining, very small dataset size with limited initial annotations, potential for co-occurring diseases, and slide imperfections.
*   **Key Requirements (MSX):** High accuracy, reliability, explainability (XAI), use of pre-trained models, and methodologies backed by scientific literature.

## PatchCamelyon (PCam) Dataset

*   **Source:** Available via [HuggingFace Datasets (`1aurent/PatchCamelyon`)](https://huggingface.co/datasets/1aurent/PatchCamelyon).
*   **Task:** Binary classification of 96x96px RGB image patches from histopathologic scans of lymph node sections.
*   **Labeling:** Each patch is labeled as either containing metastatic breast cancer tissue (`True`/`1`) or not (`False`/`0`).
*   **Splits:** Provides standard 'train', 'validation', and 'test' splits.

## Technologies & Libraries Used

*   **Primary Language:** Python 3.x
*   **Core ML/DL Framework:** PyTorch, Torchvision
*   **Data Handling & Loading:** HuggingFace `datasets`, `torch.utils.data.DataLoader`
*   **Model Architectures:** `torchvision.models` (initially ResNet18)
*   **Numerical Computation:** NumPy
*   **Data Augmentation:** Torchvision `transforms`
*   **Image Handling:** Pillow (PIL)
*   **Metrics & Evaluation:** scikit-learn
*   **Plotting & Visualization:** Matplotlib, Seaborn
*   **Progress Bars:** tqdm
*   **Configuration Management:** PyYAML
*   **Environment Management:** Conda (recommended) / venv
*   **Version Control:** Git & GitHub
*   **HPC Job Submission:** Slurm (example for Compute Canada's Graham cluster)

## Project Structure

```.
├── PCamelyon.py            # Main Python script for training, evaluation, and experiments
├── configs/                # Directory for configuration files
│   └── config.yaml         # Example YAML configuration file for experiments
├── job_submission.sh       # Example Slurm script for HPC job submission
├── PCamelyon_DRAFT.ipynb   # Initial Jupyter Notebook for exploration (archival)
├── README.md               # This file
├── requirements.txt        # Pip requirements file for project dependencies for Siku HPC System
├── requirements_local.txt  # (Optional) Pip requirements for local/dev specific tools
└── experiment_outputs/     # (Generated) Default base directory for run outputs
    └── run_timestamp_jobid/ # (Generated) Example run-specific output directory
        ├── effective_config.yaml # (Generated) The actual configuration used for the run
        ├── run.log               # (Generated) Detailed log file for the run
        ├── plots/                # (Generated) Directory for saved plots
        │   ├── label_distribution.png
        │   ├── sample_images.png
        │   ├── learning_curves_baseline.png
        │   └── ...               # Other plots like confusion matrices, ROC curves
        └── models/               # (Generated) Directory for saved model checkpoints
            └── pcam_resnet18_baseline_best.pth
            └── ...               # Other model files
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:JonathanZul/PatchChamelyonProject.git
    cd PatchChamelyonProject
    ```

2.  **Create and activate a Python environment:**
    *   **Using venv:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate  # On Linux/macOS
        # .\.venv\Scripts\activate  # On Windows
        pip install -r <requirements-file>
        ```

3.  **Data:**
    The PatchCamelyon dataset will be automatically downloaded and cached by the HuggingFace `datasets` library when `PCamelyon.py` is first run. Ensure you have an internet connection for the initial download.
    *   Default cache location: `~/.cache/huggingface/datasets`
    *   For HPC environments, it's recommended to set the `HF_DATASETS_CACHE` environment variable to a directory on a shared/project filesystem (e.g., `/project/your_alloc/your_user/.cache/huggingface/datasets` or `/scratch/your_user/.cache/huggingface/datasets`). Similarly for `TORCH_HOME` (e.g., `export TORCH_HOME="/project/your_alloc/your_user/.cache/torch"`).

## Configuration

The script `PCamelyon.py` uses a YAML configuration file to manage experiment parameters. An example can be found in `configs/config.yaml`.

**Key Configuration Parameters (see `configs/config.yaml` for a full example):**

*   `hpc_mode`: (int, 0 or 1) `0` for local/interactive mode (shows plots), `1` for HPC/batch mode (saves plots).
*   `output_dir_base`: (str) Path to the base directory where a unique sub-directory for each run's outputs (logs, models, plots, effective config) will be created.
*   `device_preference`: (str) Preferred device: "cuda", "mps", or "cpu".
*   `random_seed`: (int, optional) For experiment reproducibility.
*   `batch_size`: (int) Batch size for DataLoaders.
*   `num_workers`: (int) Number of worker processes for DataLoaders.
*   `model_architecture`: (str) Name of the model architecture to use (e.g., "resnet18").
*   `baseline_learning_rate`: (float) Learning rate for the initial baseline model training.
*   `baseline_num_epochs`: (int) Number of epochs for baseline training.
*   `optimizer_type`: (str) Type of optimizer (e.g., "Adam", "SGD").
*   `optimizer_params`: (dict, optional) Additional parameters for the optimizer (e.g., `betas` for Adam, `momentum` for SGD).
*   `lr_tuning_num_epochs`: (int) Number of epochs for learning rate tuning experiments.
*   `lrs_to_test`: (list of floats) Learning rates to test during LR experiments.

The script supports overriding some configuration parameters via command-line arguments. See `python PCamelyon.py --help`. The precedence is: **CLI arguments > Config File values > Script defaults**.

A copy of the *effective configuration* used for each run is saved as `effective_config.yaml` in the run-specific output directory.

## How to Run

1.  **Prepare Configuration:**
    *   Copy or modify `configs/config.yaml` to set your desired experiment parameters.

2.  **Local Execution:**
    ```bash
    # Activate your Python environment first
    # Example: Using defaults defined in the script (if no --config is passed and args have defaults)
    # python PCamelyon.py

    # Example: Using a specific configuration file
    python PCamelyon.py --config configs/my_experiment_config.yaml

    # Example: Overriding config output directory and enabling HPC mode via CLI
    python PCamelyon.py --config configs/my_experiment_config.yaml --output_dir ./custom_run_outputs --hpc_mode 1
    ```

3.  **HPC Execution (e.g., Compute Canada's Siku cluster using Slurm):**
    *   Modify the example `job_submission.sh` Slurm script:
        *   Update your allocation account (`--account`).
        *   Update the email where you want to receive notifications regarding the job (`--mail-user`).
        *   Adjust resource requests (`--time`, `--mem`, `--cpus-per-task`, `--gpus-per-node`).
        *   Ensure paths to your environment activation, Python script, and config file are correct.
        *   Set up cache directories (`TORCH_HOME`, `HF_DATASETS_CACHE`) to point to `/project` or `/scratch`.
        *   Specify a unique output directory for the script using `--output_dir`.
    *   Submit the job:
        ```bash
        sbatch job_submission.sh
        ```
    *   Monitor job status with `sq` and check logs in the specified Slurm output file and the script's `run.log`.

## Project Phases & Current Status

This project is structured in phases:

*   **Phase 0: Setup, Familiarization, and Initial Exploration** (`COMPLETED`)
    *   Environment setup, dataset loading/inspection.
*   **Phase 1: Baseline Model Development & Refinement** (`COMPLETED`)
    *   Custom PyTorch Dataset/DataLoader.
    *   Selection & adaptation of ResNet18.
    *   Implementation of training/validation loops with logging.
    *   Detailed evaluation on the test set (metrics, confusion matrix, ROC AUC).
    *   Results logging and learning curve visualization.
    *   Saving the best model based on validation performance.
    *   Integration of YAML configuration files and robust argument parsing, optimization for HPC environment.
*   **Phase 2: Experimentation and Model Improvement** (`IN PROGRESS`)
    *   Hyperparameter Tuning (learning rate experiments ongoing).
    *   Advanced Data Augmentation (initial implementation added, experimentation pending).
    *   (Planned) Trying Different Pre-trained Architectures (e.g., EfficientNet, ViT).
    *   (Planned) Addressing Class Imbalance (conceptual discussion for PCam, practical for MSX).
*   **Phase 3: Model Interpretation and Explainability (XAI)** (`PLANNED`)
    *   Basic Saliency Maps.
    *   Class Activation Mapping (CAM / Grad-CAM).
*   **Phase 4: Project Reflection and Transfer to MSX Project** (`PLANNED`)
    *   Summarizing key learnings.
    *   Bridging PCam experiences to the specific challenges of the MSX project.
