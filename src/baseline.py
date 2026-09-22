import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import prepare_data
from model import FeedforwardNN
from train import train_model

# ---------------------------------------------------------
# Final baseline hyperparameters selected from the full
# grid search and confirmed across five random seeds.
# ---------------------------------------------------------

HIDDEN_SIZE = 128
LEARNING_RATE = 0.1
BATCH_SIZE = 32

MAX_EPOCHS = 300
PATIENCE = 15

SEED = 42

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------
# Prepare the development data
# ---------------------------------------------------------

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

# ---------------------------------------------------------
# Create the baseline model
#
# The baseline uses all seven classes from the beginning
# and does not apply any class-imbalance correction.
# ---------------------------------------------------------

model = FeedforwardNN(
    hidden_size=HIDDEN_SIZE
)

# ---------------------------------------------------------
# Train the baseline model
# ---------------------------------------------------------

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
    verbose=True
)

# ---------------------------------------------------------
# Display development results
# ---------------------------------------------------------

print("\nBaseline development result")
print("-" * 45)

print(f"Hidden neurons: {HIDDEN_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Best validation Macro-F1: {best_val_f1:.4f}")
print(f"Best epoch: {best_epoch}")

# ---------------------------------------------------------
# Plot training and validation loss
# ---------------------------------------------------------

epochs = range(1, len(train_losses) + 1)

plt.figure()

plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline Training and Validation Loss")
plt.legend()

plt.show()

# ---------------------------------------------------------
# Plot validation Macro-F1
# ---------------------------------------------------------

plt.figure()

plt.plot(
    epochs,
    val_f1_scores
)

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Baseline Validation Macro-F1")

plt.show()