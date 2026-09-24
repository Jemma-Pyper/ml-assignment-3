# This script runs the seed-1 analysis used for detailed figures
# and per-class evaluation in the report.
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

from data import prepare_data
from model import FeedforwardNN
from train import train_model

# =========================================================
# SETTINGS
# =========================================================

SEED = 1

LEARNING_RATE = 0.1
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

ACCEPTABLE_TRAIN_F1 = 0.90
OVERFIT_GAP = 0.10
MAX_HIDDEN_NEURONS = 256

# Classes ordered from least frequent to most frequent.
# Original: 4 -> 5 -> 6 -> 7 -> 3 -> 1 -> 2
# Encoded:  3 -> 4 -> 5 -> 6 -> 2 -> 0 -> 1
CLASS_ORDER = [3, 4, 5, 6, 2, 0, 1]

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

# =========================================================
# DATA
# =========================================================

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def select_active_classes(X, y, active_classes):
    """Keep active classes and remap labels to 0, 1, 2, ..."""
    mask = np.isin(y, active_classes)

    X_subset = X[mask]
    y_subset = y[mask]

    label_map = {
        class_label: new_label
        for new_label, class_label in enumerate(active_classes)
    }

    y_remapped = np.array([
        label_map[label]
        for label in y_subset
    ])

    return X_subset, y_remapped

def calculate_macro_f1(model, X, y):
    """Calculate Macro-F1 using the current output ordering."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)

    return f1_score(
        y,
        predictions.cpu().numpy(),
        average="macro",
        zero_division=0
    )

def get_predictions(model, X):
    """Return output-neuron predictions."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)

    return predictions.cpu().numpy()

def add_hidden_neuron(model):
    """Add one hidden neuron while retaining existing weights where possible."""
    old_hidden_size = model.hidden_size

    new_model = FeedforwardNN(
        input_size=model.input_size,
        hidden_size=old_hidden_size + 1,
        output_size=model.output_size
    )

    if old_hidden_size == 0:
        with torch.no_grad():
            new_model.output.bias.copy_(model.output.bias)

        return new_model

    with torch.no_grad():
        new_model.hidden.weight[:old_hidden_size].copy_(
            model.hidden.weight
        )

        new_model.hidden.bias[:old_hidden_size].copy_(
            model.hidden.bias
        )

        new_model.output.weight[:, :old_hidden_size].copy_(
            model.output.weight
        )

        new_model.output.bias.copy_(
            model.output.bias
        )

    return new_model

def add_output_neuron(model):
    """Add one output neuron when a new class is introduced."""
    old_output_size = model.output_size

    new_model = FeedforwardNN(
        input_size=model.input_size,
        hidden_size=model.hidden_size,
        output_size=old_output_size + 1
    )

    with torch.no_grad():
        if model.hidden_size == 0:
            new_model.output.weight[:old_output_size].copy_(
                model.output.weight
            )

        else:
            new_model.hidden.weight.copy_(
                model.hidden.weight
            )

            new_model.hidden.bias.copy_(
                model.hidden.bias
            )

            new_model.output.weight[:old_output_size].copy_(
                model.output.weight
            )

        new_model.output.bias[:old_output_size].copy_(
            model.output.bias
        )

    return new_model

# =========================================================
# INCREMENTAL CLASS LEARNING
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

active_classes = CLASS_ORDER[:2]

model = FeedforwardNN(
    input_size=54,
    hidden_size=0,
    output_size=2
)

stage_results = []

while True:
    original_labels = [
        class_label + 1
        for class_label in active_classes
    ]

    print("\n" + "=" * 70)
    print(f"Training with classes: {original_labels}")
    print("=" * 70)

    X_train_active, y_train_active = select_active_classes(
        X_train,
        y_train,
        active_classes
    )

    X_val_active, y_val_active = select_active_classes(
        X_val,
        y_val,
        active_classes
    )

    stopping_reason = None

    previous_model = None
    previous_train_f1 = None
    previous_val_f1 = None
    previous_best_epoch = None

    # -----------------------------------------------------
    # Grow the hidden layer when the current model underfits.
    # -----------------------------------------------------

    while True:
        print(f"Hidden neurons: {model.hidden_size}")

        train_losses, val_losses, val_f1_scores, best_val_f1, best_epoch = train_model(
            model,
            X_train_active,
            y_train_active,
            X_val_active,
            y_val_active,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            epochs=MAX_EPOCHS,
            patience=PATIENCE,
            verbose=False
        )

        train_f1 = calculate_macro_f1(
            model,
            X_train_active,
            y_train_active
        )

        val_f1 = calculate_macro_f1(
            model,
            X_val_active,
            y_val_active
        )

        print(f"Training Macro-F1: {train_f1:.4f}")
        print(f"Validation Macro-F1: {val_f1:.4f}")
        print(f"Best epoch: {best_epoch}")

        performance_gap = train_f1 - val_f1

        if performance_gap >= OVERFIT_GAP:
            stopping_reason = "overfitting"

            print(
                "Stopping hidden-layer growth: "
                "training-validation gap suggests overfitting."
            )

            if previous_model is not None:
                model = previous_model
                train_f1 = previous_train_f1
                val_f1 = previous_val_f1
                best_epoch = previous_best_epoch

                print(
                    f"Returning to previous architecture with "
                    f"{model.hidden_size} hidden neuron(s)."
                )

            break

        if train_f1 >= ACCEPTABLE_TRAIN_F1:
            stopping_reason = "acceptable training performance"

            print(
                "Stopping hidden-layer growth: "
                "training performance is acceptable."
            )

            break

        if model.hidden_size >= MAX_HIDDEN_NEURONS:
            stopping_reason = "hidden-neuron safety limit"

            print(
                "Hidden-neuron safety limit reached while "
                "the model is still underfitting."
            )

            break

        previous_model = copy.deepcopy(model)
        previous_train_f1 = train_f1
        previous_val_f1 = val_f1
        previous_best_epoch = best_epoch

        print("Model is still underfitting. Adding one hidden neuron.")

        model = add_hidden_neuron(model)

    stage_results.append({
        "classes": original_labels.copy(),
        "hidden_size": model.hidden_size,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "best_epoch": best_epoch,
        "stopping_reason": stopping_reason
    })

    print(f"Selected architecture: {model.hidden_size} hidden neuron(s)")
    print(f"Training Macro-F1: {train_f1:.4f}")
    print(f"Validation Macro-F1: {val_f1:.4f}")
    print(f"Stopping reason: {stopping_reason}")

    if stopping_reason == "hidden-neuron safety limit":
        print(
            "Incremental learning stopped because the "
            "model still requires additional capacity."
        )
        break

    if len(active_classes) == len(CLASS_ORDER):
        break

    next_class = CLASS_ORDER[len(active_classes)]
    active_classes.append(next_class)

    print(f"Adding class {next_class + 1}")

    model = add_output_neuron(model)

# =========================================================
# FINAL TEST EVALUATION
# =========================================================

# The final test set should only be evaluated if all seven
# classes were successfully introduced.
if len(active_classes) != len(CLASS_ORDER):
    raise RuntimeError(
        "Incremental training stopped before all classes were introduced."
    )

output_predictions = get_predictions(
    model,
    X_test
)

# Convert output positions back to original encoded labels.
test_predictions = np.array([
    CLASS_ORDER[prediction]
    for prediction in output_predictions
])

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

final_result = stage_results[-1]

print("\n" + "=" * 75)
print("INCREMENTAL TEST RESULTS - SEED 1")
print("=" * 75)
print(f"Hidden neurons: {final_result['hidden_size']}")
print(f"Validation Macro-F1: {final_result['val_f1']:.4f}")
print(f"Test Macro-F1: {test_macro_f1:.4f}")
print(f"Stopping reason: {final_result['stopping_reason']}")

print("\n" + "=" * 75)
print("INCREMENTAL PER-CLASS TEST RESULTS")
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
# FIGURE 1: INCREMENTAL ARCHITECTURE GROWTH
# =========================================================

stage_labels = [
    ", ".join(map(str, result["classes"]))
    for result in stage_results
]

hidden_sizes = [
    result["hidden_size"]
    for result in stage_results
]

plt.figure(figsize=(8, 5))
plt.plot(
    range(len(stage_labels)),
    hidden_sizes,
    marker="o"
)
plt.xticks(
    range(len(stage_labels)),
    stage_labels,
    rotation=30,
    ha="right"
)
plt.xlabel("Classes included")
plt.ylabel("Hidden neurons")
plt.title("Incremental Network Growth")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "incremental_growth_seed1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# FIGURE 2: PERFORMANCE ACROSS CLASS STAGES
# =========================================================

stage_train_f1 = [
    result["train_f1"]
    for result in stage_results
]

stage_val_f1 = [
    result["val_f1"]
    for result in stage_results
]

plt.figure(figsize=(8, 5))
plt.plot(
    range(len(stage_labels)),
    stage_train_f1,
    marker="o",
    label="Training Macro-F1"
)
plt.plot(
    range(len(stage_labels)),
    stage_val_f1,
    marker="o",
    label="Validation Macro-F1"
)
plt.xticks(
    range(len(stage_labels)),
    stage_labels,
    rotation=30,
    ha="right"
)
plt.ylim(0, 1.05)
plt.xlabel("Classes included")
plt.ylabel("Macro-F1")
plt.title("Incremental Performance Across Class Stages")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "incremental_stage_f1_seed1.png",
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
plt.title("Incremental Per-Class Test Performance")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "incremental_per_class_metrics_seed1.png",
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
plt.title("Incremental Test Confusion Matrix - Seed 1")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "incremental_confusion_matrix_seed1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nIncremental figures saved to the figures/ folder.")