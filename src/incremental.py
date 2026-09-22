import copy
import random

import numpy as np
import torch
from sklearn.metrics import f1_score

from data import prepare_data
from model import FeedforwardNN
from train import train_model

# =========================================================
# SETTINGS
# =========================================================

SEED = 42

# Same training settings as the baseline.
LEARNING_RATE = 0.03
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

# Incremental-learning development criteria.
ACCEPTABLE_TRAIN_F1 = 0.90
OVERFIT_GAP = 0.10

# Temporary safety limit. The original limit of 20 was too low.
MAX_HIDDEN_NEURONS = 256

# Original class order, least frequent to most frequent:
# 4 -> 5 -> 6 -> 7 -> 3 -> 1 -> 2
#
# prepare_data() converts labels 1-7 to 0-6, giving:
# 3 -> 4 -> 5 -> 6 -> 2 -> 0 -> 1
CLASS_ORDER = [3, 4, 5, 6, 2, 0, 1]

# =========================================================
# REPRODUCIBILITY AND DATA
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def select_active_classes(X, y, active_classes):
    """
    Keep only observations belonging to the active classes
    and remap their labels to 0, 1, 2, ...
    """
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
    """Calculate Macro-F1 for a trained model."""
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

def add_hidden_neuron(model):
    """
    Create a model with one additional hidden neuron,
    retaining existing weights where possible.
    """
    old_hidden_size = model.hidden_size
    new_hidden_size = old_hidden_size + 1

    new_model = FeedforwardNN(
        input_size=model.input_size,
        hidden_size=new_hidden_size,
        output_size=model.output_size
    )

    # Moving from no hidden layer to one hidden neuron changes
    # the architecture completely, so only output biases can
    # be retained.
    if old_hidden_size == 0:
        with torch.no_grad():
            new_model.output.bias.copy_(model.output.bias)

        return new_model

    # Otherwise copy all existing hidden neurons and their
    # connections into the larger network.
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
    """
    Add one output neuron when a new class is introduced,
    retaining existing weights where possible.
    """
    old_output_size = model.output_size
    new_output_size = old_output_size + 1

    new_model = FeedforwardNN(
        input_size=model.input_size,
        hidden_size=model.hidden_size,
        output_size=new_output_size
    )

    with torch.no_grad():
        if model.hidden_size == 0:
            new_model.output.weight[:old_output_size].copy_(
                model.output.weight
            )
        else:
            new_model.hidden.weight.copy_(model.hidden.weight)
            new_model.hidden.bias.copy_(model.hidden.bias)

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

# Start with the two least frequent classes.
active_classes = CLASS_ORDER[:2]

# Simplest architecture: no hidden neurons and two outputs.
model = FeedforwardNN(
    input_size=54,
    hidden_size=0,
    output_size=2
)

stage_results = []

# =========================================================
# CLASS-LEARNING LOOP
# =========================================================

while True:
    print("\n" + "=" * 70)

    # Convert back to original Covertype labels for display.
    original_labels = [
        class_label + 1
        for class_label in active_classes
    ]

    print(f"Training with classes: {original_labels}")
    print("=" * 70)

    # Keep only the currently active classes.
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

    # Store the previous architecture in case a larger one
    # begins to overfit.
    previous_model = None
    previous_train_f1 = None
    previous_val_f1 = None
    previous_best_epoch = None

    # -----------------------------------------------------
    # Hidden-layer growth
    # -----------------------------------------------------

    while True:
        print(f"\nHidden neurons: {model.hidden_size}")

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

        # Check for overfitting.
        if performance_gap >= OVERFIT_GAP:
            stopping_reason = "overfitting"

            print(
                "Stopping hidden-layer growth: "
                "training-validation gap suggests overfitting."
            )

            # Return to the previous smaller architecture.
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

        # Check whether the model has enough capacity.
        if train_f1 >= ACCEPTABLE_TRAIN_F1:
            stopping_reason = "acceptable training performance"

            print(
                "Stopping hidden-layer growth: "
                "training performance is acceptable."
            )

            break

        # Stop completely if the temporary safety limit is
        # reached while the model is still underfitting.
        if model.hidden_size >= MAX_HIDDEN_NEURONS:
            stopping_reason = "hidden-neuron safety limit"

            print(
                "Hidden-neuron safety limit reached while "
                "the model is still underfitting."
            )

            break

        # Otherwise the model is still underfitting.
        previous_model = copy.deepcopy(model)
        previous_train_f1 = train_f1
        previous_val_f1 = val_f1
        previous_best_epoch = best_epoch

        print("Model is still underfitting. Adding one hidden neuron.")

        model = add_hidden_neuron(model)

    # -----------------------------------------------------
    # Save current class-stage result
    # -----------------------------------------------------

    stage_results.append({
        "classes": original_labels,
        "hidden_size": model.hidden_size,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "best_epoch": best_epoch,
        "stopping_reason": stopping_reason
    })

    print(
        f"\nSelected architecture: "
        f"{model.hidden_size} hidden neuron(s)"
    )
    print(f"Training Macro-F1: {train_f1:.4f}")
    print(f"Validation Macro-F1: {val_f1:.4f}")
    print(f"Stopping reason: {stopping_reason}")

    # Do not add another class if the model is still
    # underfitting at the safety limit.
    if stopping_reason == "hidden-neuron safety limit":
        print(
            "\nIncremental learning stopped because the "
            "model still requires additional capacity."
        )
        break

    # Stop once all seven classes have been introduced.
    if len(active_classes) == len(CLASS_ORDER):
        break

    # -----------------------------------------------------
    # Introduce next least frequent class
    # -----------------------------------------------------

    next_class = CLASS_ORDER[len(active_classes)]
    active_classes.append(next_class)

    print(f"\nAdding class {next_class + 1}")

    # Add one output neuron for the new class.
    model = add_output_neuron(model)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("INCREMENTAL LEARNING SUMMARY")
print("=" * 70)

for result in stage_results:
    print(
        f"Classes: {result['classes']} | "
        f"Hidden neurons: {result['hidden_size']:>2} | "
        f"Train Macro-F1: {result['train_f1']:.4f} | "
        f"Val Macro-F1: {result['val_f1']:.4f} | "
        f"Stop: {result['stopping_reason']}"
    )