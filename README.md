# Machine Learning 441 Assignment 3

## Project Overview

This project investigates whether incremental class learning improves performance on an imbalanced multi-class classification problem. The experiment compares:

1. A baseline feedforward neural network trained on all seven classes at once.
2. An incremental class learning neural network that introduces classes from least frequent to most frequent.

The dataset used is the Forest Covertype dataset from `sklearn.datasets.fetch_covtype`.

## Repository Structure

- `src/data.py` - loads, samples, splits, and standardises the dataset.
- `src/model.py` - defines the feedforward neural network.
- `src/train.py` - contains the shared training loop and early stopping.
- `src/baseline.py` - seed-1 baseline run used to generate figures.
- `src/incremental.py` - seed-1 incremental run used to generate figures.
- `src/baseline_5seed.py` - final five-seed baseline evaluation.
- `src/incremental_5seed.py` - final five-seed incremental evaluation.
- `src/grid_check.py` - compares the baseline hyperparameter settings.
- `figures/` - generated plots used in the analysis.
- `DECISIONS.md` - notes on methodological choices made during development.

## Data Preparation

The Forest Covertype dataset contains 54 input features and 7 target classes. A stratified sample of 50,000 observations is used. The data is split into 70% training, 15% validation, and 15% test sets.

The first 10 continuous features are standardised using the training set only. The remaining 44 binary wilderness and soil features are left unchanged. Class labels are converted from 1-7 to 0-6 for use in PyTorch.

## Methods

The baseline model is a feedforward neural network with one hidden layer, ReLU activation, cross-entropy loss, mini-batch SGD, and early stopping based on validation Macro-F1.

Final baseline settings:

- Hidden neurons: 128
- Learning rate: 0.1
- Batch size: 32
- Maximum epochs: 300
- Early-stopping patience: 15

The incremental model introduces classes from least frequent to most frequent:

```text
4, 5, 6, 7, 3, 1, 2
```

Training starts with classes 4 and 5. The remaining classes are then introduced one at a time in the order above. The output layer is expanded as new classes are introduced, while hidden-layer capacity can also be increased when needed.

The incremental approach uses the same general training setup as the baseline so that the main difference between the two experiments is the way classes are introduced during training.

## Evaluation

Macro-F1 is used as the main evaluation metric because the dataset is imbalanced and performance on all seven classes is important.

The final baseline and incremental models are each run using five random seeds:

```text
1, 2, 3, 4, 5
```

The five-seed experiments are used to compare the average Macro-F1 and variation in performance between the two approaches.

The seed-1 runs are also used to generate the figures for the report, including training curves, per-class performance, and confusion matrices.

## Running the Code

The scripts can be run from the project root. For example:

```bash
python src/baseline.py
python src/incremental.py
python src/baseline_5seed.py
python src/incremental_5seed.py
```

The Forest Covertype dataset is downloaded through scikit-learn and is not stored directly in the repository.

## Requirements

The main Python libraries used are:

- PyTorch
- scikit-learn
- NumPy
- pandas
- Matplotlib

The exact package requirements are listed in `requirements.txt`.