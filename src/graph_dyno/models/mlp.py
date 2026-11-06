from torch import nn
from torch.nn import functional as F
from collections.abc import Callable


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
        activation: nn.Module = nn.ReLU,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
        layer = nn.Linear(input_size, hidden_size)
        self.layers.append(layer)
        self.layers.append(self.activation())

        for _ in range(num_hidden_layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.activation())

        # Two extra activations before output for more expressivity
        self.layers.append(self.activation())
        self.layers.append(self.activation())
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
