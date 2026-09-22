import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

def create_dataloader(X, y, batch_size=64, shuffle=True):
    """
    Convert the NumPy arrays into PyTorch tensors and create
    a DataLoader so that the network can train in mini-batches.
    """

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=0.01,
    batch_size=64,       
    epochs=200,          
    patience=15,        
    verbose=True
):
    """
    Train a neural network using mini-batch stochastic
    gradient descent.

    Training loss, validation loss and validation Macro-F1
    are stored after each epoch.

    Early stopping is based on validation Macro-F1.
    """

    # Create shuffled mini-batches for model training.
    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    # Validation data does not need to be shuffled because
    # model weights are not updated during validation.
    val_loader = create_dataloader(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False
    )

    # Cross-entropy loss is used for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Mini-batch SGD is used to optimise the model weights.
    # The same optimiser will later be used for both models
    # so that the comparison remains consistent.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # Store performance measures from each epoch.
    train_losses = []
    val_losses = []
    val_f1_scores = []

    # ---------------------------------------------------------
    # Early stopping setup
    # ---------------------------------------------------------

    # Start below the possible Macro-F1 range so that the
    # first validation score will always count as an improvement.
    best_val_f1 = -1.0

    # Store the epoch where the best validation Macro-F1 occurs.
    best_epoch = 0

    # Store a copy of the model weights from the best epoch.
    best_model_state = None

    # Count how many consecutive epochs have passed without
    # improving the validation Macro-F1.
    epochs_without_improvement = 0

    for epoch in range(epochs):

        # -------------------------
        # Training phase
        # -------------------------

        model.train()

        total_train_loss = 0

        for X_batch, y_batch in train_loader:

            # Remove gradients from the previous mini-batch.
            optimizer.zero_grad()

            # Forward pass through the network.
            outputs = model(X_batch)

            # Calculate classification loss.
            loss = criterion(outputs, y_batch)

            # Calculate gradients using backpropagation.
            loss.backward()

            # Update the network weights.
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        average_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(average_train_loss)

        # -------------------------
        # Validation phase
        # -------------------------

        model.eval()

        total_val_loss = 0

        # Store predicted and true classes so that
        # validation Macro-F1 can be calculated.
        val_predictions = []
        val_targets = []

        # Gradients are not required during validation.
        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                outputs = model(X_batch)

                loss = criterion(outputs, y_batch)

                total_val_loss += loss.item() * X_batch.size(0)

                # Predicted class is the output with the highest score.
                predictions = torch.argmax(outputs, dim=1)

                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        average_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)

        # Macro-F1 gives equal importance to each class,
        # which is useful because Covertype is highly imbalanced.
        val_macro_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macro",
            zero_division=0
        )

        val_f1_scores.append(val_macro_f1)

        # -------------------------
        # Early stopping check
        # -------------------------

        # If validation Macro-F1 improves, save the current
        # model weights and reset the patience counter.
        if val_macro_f1 > best_val_f1:

            best_val_f1 = val_macro_f1
            best_epoch = epoch + 1

            # deepcopy is used so that the saved weights do not
            # keep changing as the model continues training.
            best_model_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:

            # No new best validation score was achieved.
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {average_train_loss:.4f} | "
                f"Val Loss: {average_val_loss:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f}"
            )

        # Stop training if validation Macro-F1 has not improved
        # for the specified number of consecutive epochs.
        if epochs_without_improvement >= patience:

            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best validation Macro-F1 was {best_val_f1:.4f} "
                    f"at epoch {best_epoch}."
                )

            break

    # Restore the model weights from the epoch that produced
    # the highest validation Macro-F1.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return (
        train_losses,
        val_losses,
        val_f1_scores,
        best_val_f1,
        best_epoch
    )