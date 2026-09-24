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

HIDDEN_SIZE = 128
LEARNING_RATE = 0.1
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

# Keep the same train, validation and test split for every seed.
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()


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


results = []

for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"BASELINE RUN - SEED {seed}")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = FeedforwardNN(
        hidden_size=HIDDEN_SIZE
    )

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

    test_f1 = calculate_macro_f1(
        model,
        X_test,
        y_test
    )

    results.append({
        "seed": seed,
        "val_f1": best_val_f1,
        "test_f1": test_f1,
        "best_epoch": best_epoch
    })

    print(f"Validation Macro-F1: {best_val_f1:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")
    print(f"Best epoch: {best_epoch}")


# =========================================================
# FINAL SUMMARY
# =========================================================

test_scores = [
    result["test_f1"]
    for result in results
]

print("\n" + "=" * 75)
print("FINAL BASELINE TEST RESULTS")
print("=" * 75)

for result in results:
    print(
        f"Seed {result['seed']} | "
        f"Val Macro-F1: {result['val_f1']:.4f} | "
        f"Test Macro-F1: {result['test_f1']:.4f} | "
        f"Best epoch: {result['best_epoch']}"
    )

print("-" * 75)
print(f"Mean test Macro-F1: {np.mean(test_scores):.4f}")
print(f"Std test Macro-F1:  {np.std(test_scores, ddof=1):.4f}")