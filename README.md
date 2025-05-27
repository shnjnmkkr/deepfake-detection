# DeepFake Detection Pipeline

This project implements a deep learning pipeline for detecting deepfake videos using a hybrid CNN-Transformer architecture.

## Architecture

The model combines:
- ResNet50 CNN backbone for frame feature extraction
- Vision Transformer for temporal relationship modeling
- Temporal positional encoding
- MLP classifier for final prediction

## Features

- Extracts 15 frames per video
- Optional Real-ESRGAN super-resolution for frame enhancement
- Hybrid CNN-Transformer architecture
- Comprehensive metrics logging (accuracy, precision, recall, F1, AUC)
- Confusion matrix and ROC curve visualization

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Each subfolder should contain video files (.mp4, .avi, or .mov).

## Usage

1. Prepare your dataset according to the structure above.

2. Train the model:
```bash
python train.py
```

The training script will:
- Extract frames from videos
- Apply optional super-resolution
- Train the model
- Save the best model based on AUC score
- Generate evaluation metrics and plots

## Model Outputs

The training process generates:
- Best model weights (`outputs/best_model.pth`)
- Confusion matrix plot (`outputs/confusion_matrix.png`)
- ROC curve plot (`outputs/roc_curve.png`)

## Configuration

You can modify the following parameters in `train.py`:
- `BATCH_SIZE`: Batch size for training (default: 8)
- `LEARNING_RATE`: Learning rate for Adam optimizer (default: 1e-4)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `USE_SR`: Enable/disable super-resolution (default: False)

## Model Architecture Details

1. **Frame Processing**:
   - Extracts 15 frames per video
   - Optional Real-ESRGAN super-resolution
   - Resizes frames to 224x224
   - Normalizes using ImageNet statistics

2. **Feature Extraction**:
   - Uses pretrained ResNet50 as backbone
   - Removes final classification layer
   - Extracts 2048-dimensional features per frame

3. **Temporal Modeling**:
   - Adds temporal positional encoding
   - Uses 3-layer Transformer encoder
   - 8 attention heads
   - 2048-dimensional feedforward network

4. **Classification**:
   - MLP with 512 and 128 hidden units
   - Dropout for regularization
   - Sigmoid activation for binary classification

## Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion Matrix 