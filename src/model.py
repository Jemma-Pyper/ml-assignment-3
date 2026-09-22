import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    """
    Simple feedforward neural network.

    If hidden_size = 0, the model connects the input layer
    directly to the output layer.

    If hidden_size > 0, the model uses one hidden layer
    followed by a ReLU activation.
    """

    def __init__(
        self,
        input_size=54,
        hidden_size=16,
        output_size=7
    ):
        super().__init__()

        # Store these values so that the incremental model
        # can refer to the current architecture later.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # -------------------------------------------------
        # No hidden layer
        # -------------------------------------------------

        if hidden_size == 0:

            self.hidden = None
            self.relu = None

            # Inputs connect directly to the output neurons.
            self.output = nn.Linear(
                input_size,
                output_size
            )

        # -------------------------------------------------
        # One hidden layer
        # -------------------------------------------------

        else:

            self.hidden = nn.Linear(
                input_size,
                hidden_size
            )

            self.relu = nn.ReLU()

            self.output = nn.Linear(
                hidden_size,
                output_size
            )

    def forward(self, x):

        # If there is no hidden layer, send the inputs
        # directly to the output layer.
        if self.hidden is None:
            return self.output(x)

        # Otherwise use the hidden layer and ReLU.
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)

        return x