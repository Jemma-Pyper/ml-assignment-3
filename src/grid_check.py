import random
import numpy as np
import torch

from data import prepare_data
from model import FeedforwardNN
from train import train_model

# =========================================================
# DATA
# =========================================================

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

# =========================================================
# CONFIGURATIONS TO COMPARE
# =========================================================

# Top configurations from the full grid search, plus the
# previous baseline configuration for comparison.
configs = [
    {
        "hidden_size": 128,
        "learning_rate": 0.1,
        "batch_size": 32
    },
    {
        "hidden_size": 128,
        "learning_rate": 0.1,
        "batch_size": 64
    },
    {
        "hidden_size": 128,
        "learning_rate": 0.03,
        "batch_size": 32
    },
    {
        "hidden_size": 64,
        "learning_rate": 0.03,
        "batch_size": 32
    }
]

seeds = [1, 2, 3, 4, 5]

results = []

# =========================================================
# REPEATED-RUN COMPARISON
# =========================================================

for config_number, config in enumerate(configs, start=1):

    print("\n" + "=" * 70)

    print(
        f"Configuration {config_number} | "
        f"Hidden: {config['hidden_size']} | "
        f"LR: {config['learning_rate']} | "
        f"Batch: {config['batch_size']}"
    )

    print("=" * 70)

    scores = []
    best_epochs = []

    for seed in seeds:

        # Reset randomness before every model run.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create a completely new model.
        model = FeedforwardNN(
            hidden_size=config["hidden_size"]
        )

        train_losses, val_losses, val_f1_scores, best_val_f1, best_epoch = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            epochs=300,
            patience=15,
            verbose=False
        )

        scores.append(best_val_f1)
        best_epochs.append(best_epoch)

        print(
            f"Seed {seed} | "
            f"Macro-F1: {best_val_f1:.4f} | "
            f"Best epoch: {best_epoch}"
        )

    # Summarise this configuration across the five seeds.
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores, ddof=1)
    mean_epoch = np.mean(best_epochs)

    results.append({
        "hidden_size": config["hidden_size"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_epoch": mean_epoch
    })

# =========================================================
# FINAL SUMMARY
# =========================================================

results.sort(
    key=lambda result: result["mean_f1"],
    reverse=True
)

print("\n" + "=" * 75)
print("REPEATED CONFIGURATION COMPARISON")
print("=" * 75)

for rank, result in enumerate(results, start=1):

    print(
        f"{rank}. "
        f"Hidden: {result['hidden_size']:>3} | "
        f"LR: {result['learning_rate']:<4} | "
        f"Batch: {result['batch_size']:>3} | "
        f"Mean Macro-F1: {result['mean_f1']:.4f} | "
        f"Std: {result['std_f1']:.4f} | "
        f"Mean best epoch: {result['mean_epoch']:.1f}"
    )