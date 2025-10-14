# Multilabel Baldness Classification

## Overview

This repository presents an academic implementation of a multilabel classification model for androgenetic alopecia (male pattern baldness) using deep learning techniques in PyTorch and MaxViT-T architecture. The project leverages image data and regression-based target encoding to predict the severity of baldness in scalp images, focusing on multiple levels simultaneously. The work is inspired by and extends the open dataset from uze (2024): [hair-loss Classification Model](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#).

## Contents

- `notebook.ipynb`: Main Jupyter Notebook containing data exploration, preprocessing, model architecture, training, and evaluation.
- `data/`: Directory containing training, validation, and test images and CSV label files.
- `Model parameters results.xlsx`: Summary of model hyperparameters and outcomes.
- `maxvit_t_model_state_dict.pth`: Saved state dict of the trained MaxViT-T model.
- `.gitattributes`: Git attributes for repository management.

## Dataset

- **Source**: [Roboflow Universe - Hair Loss Classification](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#)
- **Structure**: Images are labeled for seven levels of baldness (`LEVEL_2` to `LEVEL_7`), stored in CSV files for `train`, `valid`, and `test` splits.
- **Size**:
  - Train: 1,294 samples
  - Validation: 133 samples
  - Test: 67 samples
- **No missing values** were detected in any split.

## Methodology

### Data Exploration

- Initial exploration confirms balanced label distribution in the training set, with validation and test sets showing some class imbalance.
- Images have been pre-augmented (horizontal and vertical flips).

### Preprocessing

- One-hot encoded labels are transformed into a regression target on a scale from 0 (least severe) to 1 (most severe).
- Images are resized to 224x224 and stacked into PyTorch tensors.

### Data Augmentation

- Training images are further augmented with random rotations, resized crops, equalization, saturation, hue, contrast, and brightness adjustments.
- Normalization follows ImageNet standards for compatibility with pretrained models.

### Model Architecture

- **Backbone**: MaxViT-T pretrained on ImageNet.
- **Modifications**:
  - First block weights are frozen.
  - Dropout layers (except the frozen block) are set to 0.5.
  - The final classifier layer is replaced to output a single regression value.

### Training

- Loss function: L1 loss for regression.
- Optimizer: AdamW with learning rate scheduling and weight decay.
- Training incorporates mixed precision (autocast) and optional Exponential Moving Average (EMA) smoothing.
- Early stopping is implemented to prevent overfitting.

### Evaluation

- Training progress is monitored using validation loss.
- Model parameters and results are summarized in the provided `.xlsx` file.

## Results

- The model achieves substantial reduction in validation loss over training epochs, indicating effective learning.
- Model weights and training logs are available for reproducibility.

## Reproducibility

To reproduce the results:

1. Place the image data and CSV label files in the `data/` directory as per the notebook instructions.
2. Run `notebook.ipynb` in a Python environment with the required packages: `torch`, `torchvision`, `ema_pytorch`, `kornia`, `matplotlib`, `seaborn`, `pandas`, `numpy`, and `PIL`.
3. Model states and processed tensors are saved for later use.

## References

- uze (2024). Hair-loss Classification Model. [Roboflow Universe](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#).
- MaxViT: Google Research, Image Classification Models.

## License

This repository and its contents are intended for academic research and educational purposes.
