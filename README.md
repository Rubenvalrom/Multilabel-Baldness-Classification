# Multilabel Baldness Classification

## Overview

This repository presents an academic implementation of a multilabel classification model for androgenetic alopecia (male pattern baldness) using deep learning techniques in PyTorch and MaxViT-T architecture. The project leverages image data and regression-based target encoding to predict the severity of baldness in scalp images, focusing on multiple levels simultaneously. The work is inspired by and extends the open dataset from uze (2024): [hair-loss Classification Model](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#).

## Contents
- `src/notebook.ipynb`: Main Jupyter Notebook with EDA, data preprocessing, model architecture, training, and evaluation with L1Loss, which gave the better generalization.
- `src/losses notebooks/`: Jupyter Notebook containing data preprocessing, model architecture, training, and evaluation of CrossEntropy, Coral and Corn Losses.
- `src/data/`: Directory containing training, validation, and test images along with CSV label files.
- `src/models/`: Directory containing the `AlopeciaClassifier` model and trained model state dictionaries in `state_dicts/` subdirectory.
- `src/main.py`: Prediction script for image-based severity prediction using the trained model.

## Dataset

- **Source**: [Roboflow Universe - Hair Loss Classification](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#)
- **Structure**: Images are labeled for seven levels of baldness (`LEVEL_2` to `LEVEL_7`), stored in CSV files for `train`, `valid`, and `test` splits.
- **Size**:
  - Train: 1,294 samples (It was already originally augmented, around 400 distinct subjects)
  - Validation: 133 samples
  - Test: 67 samples
- **Preprocessing**:
  - Resize to 224x244 for memory purposes and MaxVit's requirements. 
  - Rotation, Brightness, Contrast, Hue, Saturation, Gaussian noise and Angle distorsion augmentations are applied randomly during training, varying each epoch.

## Methodology

### Data Exploration

- Visual inspection and statistical summaries confirm balanced label distribution in the training set. Validation and test sets show some minor class imbalance.
- Images were pre-augmented (horizontal and vertical flips).

### Preprocessing

- One-hot encoded labels are transformed into a regression target on a scale from 0 (least severe) to 1 (most severe).
- Images are resized to 224x224 and stacked into PyTorch tensors using the `torchvision` package.

### Data Augmentation

- Training images are augmented with random rotations, resized crops, equalization, saturation, hue, contrast, and brightness adjustments.
- Normalization follows ImageNet standards for compatibility with pretrained models.

### Model Architecture

- **Backbone**: MaxViT-T pretrained on ImageNet 1K.
- **Modifications**:
  - First block weights are frozen.
  - Dropout layers (except the frozen block) are set to 0.5.
  - The final classifier layer is replaced to output a single regression value with a sigmoid activation function.

### Training

- **Loss Function**: L1 loss(MAE) for regression.
- **Optimizer**: AdamW with scheduled learning rate, momentum adjustments, and weight decay.
- Mixed precision training (using `torch.cuda.amp`) which drastically reduces memory consumption.
- Early stopping with patience of 50 epochs is implemented to improve generalization.

### Evaluation

- Bootstrap confidence intervals are reported for all sets performances.
- Accuracy, MAE, Mape and Quadratic Cohen's Kappa score.
- Confusion Matrix and Distribution of MAE on the whole set, pre-boostraping.

## Results

- Test set:
  - Accuracy: 0.75 ± 0.05,
  - MAE: 0.31 ± 0.05 
  - Mean Absolute Percentage Error: 11.40 ± 2.40
  - Quadratic Cohen's Kappa: 0.95 ± 0.02
    
- Validation set:
  - Accuracy: 0.74 ± 0.04
  - Mean Absolute Error: 0.32 ± 0.05
  - Mean Absolute Percentage Error: 12.92 ± 3.02
  - Quadratic Cohen's Kappa: 0.92 ± 0.02 
    
- Train set got all metrics effectively perfect even after the online augmentation and 0.5 dropout rate.

Overall the model has a good performance, it usually classifies correctly each subject, in a minority of cases it over/underestimates by 1 class the severity of the alopecia.

## Usage

### Prediction Script

The repository includes a `main.py` script that allows you to predict androgenetic alopecia severity for individual images using the trained `AlopeciaClassifier` model.

**How to use:**
1. Run the script: `python src/main.py`
2. A file dialog will open - select an image file (supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP, AVIF)
3. The image will be displayed, and the predicted severity level will be printed

The `AlopeciaClassifier` model automatically loads the trained weights from `src/models/state_dicts/maxvit_t_model_state_dict.pth` and applies the necessary preprocessing transformations (resize to 224x224 and ImageNet normalization).

## References

1. “hair-loss Classification Model by uze,” Roboflow, Jun. 19, 2024. https://universe.roboflow.com/uze/hair-loss-nq8hh.
2. Z. Tu et al., “MaxVIT: Multi-Axis Vision Transformer,” arXiv.org, Apr. 04, 2022. https://arxiv.org/abs/2204.01697.
3. X. Shi, W. Cao, and S. Raschka, “Deep neural networks for rank-consistent ordinal regression based on conditional probabilities,” Pattern Analysis and Applications, vol. 26, no. 3, pp. 941–955, Jun. 2023, doi: 10.1007/s10044-023-01181-9.

## License

This repository and its contents are licensed under the CC BY 4.0 License. For more details, see the `LICENSE` file.
