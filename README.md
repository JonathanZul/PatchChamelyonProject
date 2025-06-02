# PatchCamelyon Histology Image Classification: A Learning Project

## Overview

This repository documents a machine learning project focused on binary image classification using the **PatchCamelyon (PCam)** dataset. This project serves as a practical learning exercise to understand and implement a complete ML/DL pipeline for histology image analysis.

The primary motivation is to build foundational skills and explore techniques that will be transferable to a future, real-world biomedical challenge: **the detection of *Haplosporidium nelsoni* (MSX) disease in high-resolution histology slide images (WSIs) of oysters.**

## Learning Objectives (for this PCam Project)

This "dummy" project aims to achieve the following learning objectives:

*   Understand and implement a complete ML/DL image classification pipeline.
*   Learn effective data loading, preprocessing, and augmentation for histology image patches.
*   Gain experience with selecting, fine-tuning, and evaluating pre-trained DL models (e.g., ResNet, EfficientNet, Vision Transformers).
*   Explore techniques for dealing with potentially imbalanced datasets.
*   Practice implementing basic model interpretation/explainability techniques (XAI).
*   Learn to structure code modularly and document experiments effectively.
*   Simulate working with a binary classification task relevant to "disease presence/absence."

## The Target MSX Oyster Disease Project (Context)

The ultimate goal this project prepares for involves:

*   **Goal:** Implement an ML/DL system to assist experts in identifying MSX presence in oyster histology slices.
*   **Desired Output:** Signal MSX presence, ideally with bounding boxes around affected areas.
*   **Anticipated Dataset Characteristics (MSX):**
    *   Large WSI dimensions requiring patching.
    *   H&E-like staining.
    *   Very small dataset size with limited initial annotations.
    *   Potential for other co-occurring diseases.
    *   Slide imperfections (missing tissue, artifacts).
*   **Key Requirements (MSX):** High accuracy, reliability, explainability (XAI), use of pre-trained models, and methodologies backed by scientific literature.

## PatchCamelyon (PCam) Dataset

*   **Source:** Available via [HuggingFace Datasets (`patch_camelyon`)]([https://huggingface.co/datasets/patch_camelyon](https://huggingface.co/datasets/1aurent/PatchCamelyon)).
*   **Task:** Binary classification of 96x96px RGB image patches from histopathologic scans of lymph node sections.
*   **Labeling:** Each patch is labeled as either containing metastatic breast cancer tissue (`True`/`1`) or not (`False`/`0`).
*   **Splits:** Provides standard 'train', 'validation', and 'test' splits.

## Technologies & Libraries Used

*   **Primary Language:** Python 3.x
*   **TBC**
## Project Structure (TBC)

_(This will be adjusted once refined)._

## Setup and Installation

**TBC**

## How to Run

**TBC**

## Project Phases & Current Status

This project is structured in phases:

*   **Phase 0: Setup, Familiarization, and Initial Exploration** (`COMPLETED`)
    *   Environment setup, dataset loading/inspection.
*   **Phase 1: Baseline Model Development** (`COMPLETED`)
    *   Custom PyTorch Dataset/DataLoader.
    *   Selection & adaptation of ResNet18.
    *   Implementation of training/validation loops.
    *   Detailed evaluation on the test set (metrics, confusion matrix, ROC AUC).
    *   Results logging and learning curve visualization.
    *   Saving the best model based on validation performance.
*   **Phase 2: Experimentation and Model Improvement** (`IN PROGRESS`)
    *   Hyperparameter Tuning (e.g., learning rate, batch size).
    *   Advanced Data Augmentation.
    *   Trying Different Pre-trained Architectures (e.g., EfficientNet, ViT).
    *   Addressing Class Imbalance (conceptual discussion for PCam, practical for MSX).
*   **Phase 3: Model Interpretation and Explainability (XAI)** (`PLANNED`)
    *   Basic Saliency Maps.
    *   Class Activation Mapping (CAM / Grad-CAM).
*   **Phase 4: Project Reflection and Transfer to MSX Project** (`PLANNED`)
    *   Summarizing key learnings.
    *   Bridging PCam experiences to the specific challenges of the MSX project.
