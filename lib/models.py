# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/aleximmer/heteroscedastic-nn
# Copyright (c) 2025 Alex Immer, licensed under the MIT License
# License is provided for attribution purposes only, Not a Contribution

"""Implementation of multi-layer perceptron for regression."""

from torch import nn


def get_activation(act_str: str):
    """Map activation name to torch callable."""
    if act_str == "relu":
        return nn.ReLU
    if act_str == "tanh":
        return nn.Tanh
    if act_str == "selu":
        return nn.SELU
    if act_str == "silu":
        return nn.SiLU
    if act_str == "gelu":
        return nn.GELU
    else:
        raise ValueError("invalid activation")


class MLP(nn.Sequential):
    """Implement a simple MLP."""

    def __init__(
        self,
        input_size: int,
        width: int,
        depth: int,
        output_size: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        """Initialize MLP with specific width and depth."""
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)
        self.rep_layer = f"layer{depth}"
        self.add_module("flatten", nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # We have a linear model.
            self.add_module("lin_layer", nn.Linear(self.input_size, output_size, bias=True))
        else:
            # Create the MLP.
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f"layer{i+1}", nn.Linear(in_size, out_size, bias=True))
                if dropout > 0.0:
                    self.add_module(f"dropout{i+1}", nn.Dropout(p=dropout))
                self.add_module(f"{activation}{i+1}", act())
            self.add_module("out_layer", nn.Linear(hidden_sizes[-1], output_size, bias=True))
