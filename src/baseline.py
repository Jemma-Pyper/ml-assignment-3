from pathlib import Path
from data import prepare_data
from model import FeedforwardNN
from train import train_model

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

# This script runs the seed-1 analysis used for detailed figures
# and per-class evaluation in the report.import random

# =========================================================
# SETTINGS
# =========================================================

SEED = 1

HIDDEN_SIZE = 128
LEARNING_RATE = 0.1
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

# =========================================================
# DATA
# =========================================================

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

# =========================================================
# HELPER FUNCTION
# =========================================================

def get_predictions(model, X):
    """Return predicted class labels."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)

    return predictions.cpu().numpy()

# =========================================================
# TRAIN BASELINE MODEL
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model = FeedforwardNN(hidden_size=HIDDEN_SIZE)

train_losses, val_losses, val_f1_scores, best_val_f1, best_epoch = train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    patience=PATIENCE,
    verbose=False
)

# =========================================================
# TEST EVALUATION
# =========================================================

test_predictions = get_predictions(model, X_test)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

precision, recall, class_f1, support = precision_recall_fscore_support(
    y_test,
    test_predictions,
    labels=np.arange(7),
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=np.arange(7)
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 75)
print("BASELINE TEST RESULTS - SEED 1")
print("=" * 75)
print(f"Validation Macro-F1: {best_val_f1:.4f}")
print(f"Test Macro-F1: {test_macro_f1:.4f}")
print(f"Best epoch: {best_epoch}")

print("\n" + "=" * 75)
print("BASELINE PER-CLASS TEST RESULTS")
print("=" * 75)

for class_index in range(7):
    print(
        f"Class {class_index + 1} | "
        f"Precision: {precision[class_index]:.4f} | "
        f"Recall: {recall[class_index]:.4f} | "
        f"F1: {class_f1[class_index]:.4f} | "
        f"Support: {support[class_index]}"
    )

# =========================================================
# FIGURE 1: TRAINING AND VALIDATION LOSS
# =========================================================

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs, train_losses, label="Training loss")
plt.plot(epochs, val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Baseline Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "baseline_loss_seed1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# FIGURE 2: VALIDATION MACRO-F1
# =========================================================

plt.figure(figsize=(7, 5))
plt.plot(epochs, val_f1_scores)
plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best epoch ({best_epoch})"
)
plt.xlabel("Epoch")
plt.ylabel("Validation Macro-F1")
plt.title("Baseline Validation Macro-F1")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "baseline_validation_f1_seed1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# FIGURE 3: PER-CLASS PERFORMANCE
# =========================================================

classes = np.arange(1, 8)
x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, class_f1, width, label="F1")
plt.xticks(x, classes)
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Score")
plt.title("Baseline Per-Class Test Performance")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "baseline_per_class_metrics_seed1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# FIGURE 4: CONFUSION MATRIX
# =========================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(1, 8)
)

display.plot()
plt.title("Baseline Test Confusion Matrix - Seed 1")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "baseline_confusion_matrix_seed1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nBaseline figures saved to the figures/ folder.")