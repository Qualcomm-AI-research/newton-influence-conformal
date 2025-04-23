# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""General auxiliary functions used throughout the code."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.backends import cudnn


def set_seed(seed: int):
    """Set random seeds globally."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def reset_parameters(model: nn.Module):
    """Reset parameters of linear layers."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()


def to_numpy(x: Tensor) -> NDArray:
    """Map torch tensors to numpy arrays."""
    return x.cpu().detach().numpy()
