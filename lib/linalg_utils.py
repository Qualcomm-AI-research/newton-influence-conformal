# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/AaltoML/sfr
# Copyright (c) 2024 Aidan Scannell, licensed under the MIT License
# License is provided for attribution purposes only, Not a Contribution

"""General auxiliary linear algebra functions used throughout the code."""

from laplace.utils.utils import invsqrt_precision
from torch import Tensor, nn


def cholesky_add_jitter_until_psd(
    posterior_precision: Tensor, jitter: float = 1e-4, jitter_factor: float = 4
) -> Tensor:
    """Compute the inverse of the posterior precision using Cholesky decomposition.

    If the decomposition fails, we try again by adding a small constant (`jitter`) until we can
    compute the Cholesky decomposition successfully.

    Parameters
    ----------
    posterior_precision: Tensor
        The precision matrix of the posterior distribution.
    jitter: float
        Constant to be added to the precision matrix in case the Cholesky decomposition fails.
    jitter_factor: float
        Factor by which we increase the jitter each time the Cholesky decomposition fails.

    Returns
    -------
    Tensor
        Inverse of the precision of matrix.
    """
    try:
        posterior_scale = invsqrt_precision(posterior_precision)
        return posterior_scale
    except RuntimeError:
        print(f"Cholesky failed so adding more jitter={jitter}")
        jitter = jitter_factor * jitter
        posterior_precision[
            range(len(posterior_precision)), range(len(posterior_precision))
        ] += jitter
        return cholesky_add_jitter_until_psd(posterior_precision, jitter=jitter)
