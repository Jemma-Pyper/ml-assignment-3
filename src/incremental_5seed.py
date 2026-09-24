import copy
import random
import numpy as np
import torch
from sklearn.metrics import f1_score

from data import prepare_data
from model import FeedforwardNN
from train import train_model

# Final five-seed evaluation used to report mean and
# standard deviation test Macro-F1.

SEEDS = [1, 2, 3, 4, 5]

LEARNING_RATE = 0.1
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

ACCEPTABLE_TRAIN_F1 = 0.90
OVERFIT_GAP = 0.10
MAX_HIDDEN_NEURONS = 256

# Original classes from least frequent to most frequent:
# 4 -> 5 -> 6 -> 7 -> 3 -> 1 -> 2
#
# prepare_data() converts labels 1-7 to 0-6:
# 3 -> 4 -> 5 -> 6 -> 2 -> 0 -> 1
CLASS_ORDER = [3, 4, 5, 6, 2, 0, 1]

# Keep the same train, validation and test split for every seed.
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()


def select_active_classes(X, y, active_classes):
    """Keep active classes and remap their labels to 0, 1, 2, ..."""
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
    """Add one hidden neuron while retaining existing weights where possible."""
    old_hidden_size = model.hidden_size

    new_model = FeedforwardNN(
        input_size=model.input_size,
        hidden_size=old_hidden_size + 1,
        output_size=model.output_size
    )

    # Moving from no hidden layer to one hidden neuron changes
    # the architecture, so only the output biases are retained.
    if old_hidden_size == 0:
        with torch.no_grad():
            new_model.output.bias.copy_(
                model.output.bias
            )

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


all_seed_results = []

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"INCREMENTAL RUN - SEED {seed}")
    print("=" * 80)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Start with the two least frequent classes.
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

        print("\n" + "-" * 70)
        print(f"Training with classes: {original_labels}")
        print("-" * 70)

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

        # -------------------------------------------------
        # HIDDEN-LAYER GROWTH
        # -------------------------------------------------

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

            # Stop growing if the larger model begins to overfit.
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

            # Stop growing if training performance is acceptable.
            if train_f1 >= ACCEPTABLE_TRAIN_F1:
                stopping_reason = "acceptable training performance"

                print(
                    "Stopping hidden-layer growth: "
                    "training performance is acceptable."
                )

                break

            # Safety stop if the model remains underfitted.
            if model.hidden_size >= MAX_HIDDEN_NEURONS:
                stopping_reason = "hidden-neuron safety limit"

                print(
                    "Hidden-neuron safety limit reached while "
                    "the model is still underfitting."
                )

                break

            # Otherwise add one hidden neuron.
            previous_model = copy.deepcopy(model)
            previous_train_f1 = train_f1
            previous_val_f1 = val_f1
            previous_best_epoch = best_epoch

            print(
                "Model is still underfitting. "
                "Adding one hidden neuron."
            )

            model = add_hidden_neuron(model)

        # -------------------------------------------------
        # SAVE CURRENT CLASS-STAGE RESULT
        # -------------------------------------------------

        stage_results.append({
            "classes": original_labels,
            "hidden_size": model.hidden_size,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "best_epoch": best_epoch,
            "stopping_reason": stopping_reason
        })

        print(
            f"Selected architecture: "
            f"{model.hidden_size} hidden neuron(s)"
        )

        print(f"Training Macro-F1: {train_f1:.4f}")
        print(f"Validation Macro-F1: {val_f1:.4f}")
        print(f"Stopping reason: {stopping_reason}")

        # Do not introduce another class if the safety limit
        # was reached while the model was still underfitting.
        if stopping_reason == "hidden-neuron safety limit":
            print(
                "Incremental learning stopped because the "
                "model still requires additional capacity."
            )
            break

        # Stop once all seven classes have been introduced.
        if len(active_classes) == len(CLASS_ORDER):
            break

        # Introduce the next least frequent class.
        next_class = CLASS_ORDER[len(active_classes)]
        active_classes.append(next_class)

        print(f"Adding class {next_class + 1}")

        model = add_output_neuron(model)

    # -----------------------------------------------------
    # FINAL TEST EVALUATION FOR THIS SEED
    # -----------------------------------------------------

    final_result = stage_results[-1]

    if len(active_classes) == len(CLASS_ORDER):
        # Remap the test labels into the same output order
        # used by the incremental network.
        X_test_active, y_test_active = select_active_classes(
            X_test,
            y_test,
            CLASS_ORDER
        )

        test_f1 = calculate_macro_f1(
            model,
            X_test_active,
            y_test_active
        )

    else:
        test_f1 = np.nan

    all_seed_results.append({
        "seed": seed,
        "train_f1": final_result["train_f1"],
        "val_f1": final_result["val_f1"],
        "test_f1": test_f1,
        "hidden_size": final_result["hidden_size"],
        "stopping_reason": final_result["stopping_reason"]
    })

    print("\n" + "=" * 70)
    print(f"FINAL INCREMENTAL RESULT - SEED {seed}")
    print("=" * 70)

    print(f"Hidden neurons: {final_result['hidden_size']}")
    print(f"Validation Macro-F1: {final_result['val_f1']:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")
    print(f"Stopping reason: {final_result['stopping_reason']}")


# =========================================================
# FINAL SUMMARY
# =========================================================

test_scores = [
    result["test_f1"]
    for result in all_seed_results
]

print("\n" + "=" * 80)
print("FINAL INCREMENTAL TEST RESULTS")
print("=" * 80)

for result in all_seed_results:
    print(
        f"Seed {result['seed']} | "
        f"Hidden neurons: {result['hidden_size']:>3} | "
        f"Val Macro-F1: {result['val_f1']:.4f} | "
        f"Test Macro-F1: {result['test_f1']:.4f} | "
        f"Stop: {result['stopping_reason']}"
    )

print("-" * 80)
print(f"Mean test Macro-F1: {np.nanmean(test_scores):.4f}")
print(f"Std test Macro-F1:  {np.nanstd(test_scores, ddof=1):.4f}")