# Multilabel Baldness Classification

## Overview

This repository presents an academic implementation of a multilabel classification model for androgenetic alopecia (male pattern baldness) using deep learning techniques in PyTorch and MaxViT-T architecture. The project leverages image data and regression-based target encoding to predict the severity of baldness in scalp images, focusing on multiple levels simultaneously. The work is inspired by and extends the open dataset from uze (2024): [hair-loss Classification Model](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#).

## Contents

- `src/losses notebooks/classification-losses.ipynb`: Jupyter Notebook containing data exploration, preprocessing, model architecture, training, and evaluation.
- `src/data/`: Directory containing training, validation, and test images along with CSV label files.
- `src/models/`: Directory containing trained model state dictionaries like `maxvit_t_model_state_dict.pth`.

## Dataset

- **Source**: [Roboflow Universe - Hair Loss Classification](https://universe.roboflow.com/uze/hair-loss-nq8hh/dataset/1#)
- **Structure**: Images are labeled for seven levels of baldness (`LEVEL_2` to `LEVEL_7`), stored in CSV files for `train`, `valid`, and `test` splits.
- **Size**:
  - Train: 1,294 samples
  - Validation: 133 samples
  - Test: 67 samples
- **Preprocessing**: Auto-orientation, resizing to 640x640, and rotational augmentations.

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

- **Backbone**: MaxViT-T pretrained on ImageNet.
- **Modifications**:
  - First block weights are frozen.
  - Dropout layers (except the frozen block) are set to 0.5.
  - The final classifier layer is replaced to output a single regression value.

### Training

- **Loss Function**: L1 loss for regression.
- **Optimizer**: AdamW with scheduled learning rate, momentum adjustments, and weight decay.
- Mixed precision training (using `torch.cuda.amp`) accelerates convergence and reduces memory consumption.
- Early stopping with patience of 10 epochs is implemented to prevent overfitting.

### Evaluation

- Training progress is monitored using validation loss and other metrics like MAE.
- Bootstrap confidence intervals are reported for test set performance.

## Results

- The model achieves a Mean Absolute Error (MAE) of 0.12 on the validation set and 0.15 on the test set.
- Training logs, model weights, and evaluation scripts are included for reproducibility.

## Reproducibility

To reproduce the results:

1. Clone this repository: `git clone https://github.com/Rubenvalrom/Multilabel-Baldness-Classification.git`
2. Place the image data and CSV label files in the `src/data/` directory as per the notebook instructions.
3. Install dependencies using the provided `requirements.txt` file.
4. Run `src/losses notebooks/classification-losses.ipynb` in a Python environment with the required packages: `torch`, `torchvision`, `ema_pytorch`, `kornia`, `matplotlib`, `seaborn`, `pandas`, `numpy`, and `PIL`.
5. Experiment with the model using the provided datasets and evaluation scripts.

## References

1. “hair-loss Classification Model by uze,” Roboflow, Jun. 19, 2024. https://universe.roboflow.com/uze/hair-loss-nq8hh.
2. Z. Tu et al., “MaxVIT: Multi-Axis Vision Transformer,” arXiv.org, Apr. 04, 2022. https://arxiv.org/abs/2204.01697.
3. X. Shi, W. Cao, and S. Raschka, “Deep neural networks for rank-consistent ordinal regression based on conditional probabilities,” Pattern Analysis and Applications, vol. 26, no. 3, pp. 941–955, Jun. 2023, doi: 10.1007/s10044-023-01181-9.

## License

This repository and its contents are licensed under the CC BY 4.0 License. For more details, see the `LICENSE` file.
