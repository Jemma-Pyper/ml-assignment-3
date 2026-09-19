import torch
import torch.nn as nn


class FeedforwardNN(nn.Module):
    """
    Simple feedforward neural network for the Forest Covertype
    classification problem.

    The network contains:
    - 54 input features
    - 1 hidden layer
    - ReLU activation
    - 7 output neurons, one for each cover type
    """

    def __init__(self, input_size=54, hidden_size=16, output_size=7):
        super().__init__()

        # First layer connects the 54 input features to the hidden layer.
        # hidden_size is kept as a parameter because the number of hidden
        # neurons will be tuned during the baseline experiments.
        self.hidden = nn.Linear(input_size, hidden_size)

        # ReLU introduces non-linearity so that the network can learn
        # relationships that cannot be represented by a purely linear model.
        self.relu = nn.ReLU()

        # Output layer contains one neuron for each of the 7 cover types.
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define how data passes through the network.
        """

        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)

        return x