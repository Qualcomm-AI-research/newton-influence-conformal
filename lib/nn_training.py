# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/aleximmer/heteroscedastic-nn
# Copyright (c) 2025 Alex Immer, licensed under the MIT License
# License is provided for attribution purposes only, Not a Contribution

"""Regression training routines, including quantile regression."""

from __future__ import annotations

import copy
import logging
import warnings
from math import sqrt
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from laplace.utils import expand_prior_precision
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import MSELoss
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def train_regression_net(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader] = None,
    prior_prec: float = 1.0,
    n_epochs: int = 500,
    sigma_noise_posthoc: bool = True,
    optimizer_kwargs: Optional[dict] = None,
    scheduler_cls: Optional[LRScheduler] = None,
    scheduler_kwargs: Optional[dict] = None,
    eval_frequency: int = 50,
    progress_bar: bool = False,
) -> Tuple[nn.Module, Tensor, List[float], Optional[List[float]]]:
    """Train a regression model (neural network) on the data provided on train_loader.

    Parameters
    ----------
    model: nn.Module
        The model to be trained.
    train_loader: DataLoader
        Data loader containing the training data.
    valid_loader: DataLoader
        Data loader containing the validation data.
    prior_prec: float
        Initial prior precision.
    n_epochs: int
        How many epochs the model is trained for.
    sigma_noise_posthoc: bool
        Whether to estimate the observation noise after training the model.
    optimizer_kwargs: dict
        Keyword arguments of the optimizer.
    scheduler_cls: LRScheduler
        Class defining the desired learning rate scheduler.
    scheduler_kwargs: dict
        Keyword arguments of the scheduler.
    eval_frequency: int
        Define the frequency at which we evaluate the model on the validation data.
        We run the evaluation every `eval_frequency` epochs.
    progress_bar: bool
        Whether to print a progress bar to the screen.

    Returns
    -------
    model: nn.Module
        The trained model.
    sigma_noise: Tensor
        The estimated observation noise.
    losses: List[float]
        List containing the training loss for each epoch.
    valid_nlls: Optional[List[float]]
        List containing the validation log likelihood for every `eval_frequency` training epoch.
    """
    if optimizer_kwargs is not None and "weight_decay" in optimizer_kwargs:
        warnings.warn("Weight decay is handled and optimized. Will be set to 0.")
        optimizer_kwargs["weight_decay"] = 0.0

    # Get device, data set size, number of layers, number of parameters.
    device = parameters_to_vector(model.parameters()).device
    num_train = len(train_loader.dataset)
    if valid_loader is not None:
        num_val = len(valid_loader.dataset)
    # Define optimization criterion.
    criterion = MSELoss(reduction="mean")
    # Set up model optimizer.
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    params_opt = list(model.parameters())
    if not sigma_noise_posthoc:  # NB: using same optimizer setup as model training.
        log_sigma_sq_noise = nn.Parameter(-torch.ones(1, device=device))
        log_sigma_sq_noise.requires_grad = True
        params_opt += [log_sigma_sq_noise]
    else:
        log_sigma_sq_noise = None
    optimizer = Adam(params_opt, **optimizer_kwargs)
    # Set up learning rate scheduler.
    scheduler = None
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    losses = []
    valid_nlls = []
    pbar = tqdm.trange(
        1,
        n_epochs + 1,
        disable=not progress_bar,
        position=1,
        leave=False,
        desc="[Training]",
        colour="blue",
    )
    # Run optimization.
    for epoch in pbar:
        epoch_loss = 0
        epoch_perf = 0
        # Standard NN training per batch.
        for data in train_loader:
            x, y = data
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            theta = parameters_to_vector([p for p in model.parameters() if p.requires_grad])
            f = model(x)
            if not sigma_noise_posthoc:
                sigma_sq_noise = torch.exp(log_sigma_sq_noise).clamp(min=1e-4)
            else:
                sigma_sq_noise = torch.tensor(1.0, device=device)
            # For regression also rescale weight decay by n_outputs.
            loss = criterion(f, y) + ((prior_prec * sigma_sq_noise * theta) @ theta) / (
                num_train * y.shape[1]
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() * len(y)
            epoch_perf += (f.detach() - y).square().sum()
        # Update learning rate via scheduler.
        if scheduler is not None:
            scheduler.step()
        losses.append(epoch_loss / num_train)
        logging.info(
            f"[epoch={epoch}]: network training. Loss={losses[-1]:.3f}."
            + f"Perf={epoch_perf / num_train:.3f}"
        )
        # Compute negative log-likelihood on validation data.
        if valid_loader is not None:
            if (epoch % eval_frequency) == 0 or epoch == n_epochs:
                with torch.no_grad():
                    val_nll = 0
                    for x_val, y_val in valid_loader:
                        x_val, y_val = x_val.detach(), y_val.detach()
                        f_val = model(x_val)
                        log_lik = Normal(loc=f_val, scale=torch.sqrt(sigma_sq_noise)).log_prob(
                            y_val
                        )
                        val_nll += -log_lik.sum() / num_val
                    valid_nlls.append(val_nll.item())

    # ML estimate of observation noise
    # https://github.com/aleximmer/heteroscedastic-nn/blob/353575bfda555523b67e9514caff66069808b9e8/
    # run_uci_crispr_regression.py#L113
    # Could instead treat sigma_noise as param as train jointly (i.e. PNN style)
    if sigma_noise_posthoc:
        ssqe = 0
        for x, y in train_loader:
            with torch.no_grad():
                ssqe += (y - model(x)).square().sum().item() / (num_train * y.shape[1])
        sigma_noise = np.sqrt(np.clip(ssqe, 1e-4, None)).item()
    else:
        sigma_noise = sqrt(torch.exp(log_sigma_sq_noise).clamp(min=1e-4).item())
    if valid_loader is not None:
        return model, sigma_noise, losses, valid_nlls
    else:
        return model, sigma_noise, losses, None


def make_functional(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[Callable, Tuple[Tensor]]:
    """Return a functional evaluating the given model (nn.Module) on a tensor.

    This function reimplements the deprecated make_functional() from Pytorch.

    Parameters
    ----------
    model: nn.Module
        The module wrapped by the functional.
    disable_autograd_tracking: bool
        Flag to disable gradients tracking for output parameters.

    Returns
    -------
    fmodel: Callable
        The function wrapping the given Pytorch module.
    params_values: Tuple[Tensor]
        The parameter values of the wrapped Pytorch module.
    """
    params_dict = dict(model.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_model = copy.deepcopy(model)
    stateless_model.to("meta")

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        return torch.func.functional_call(stateless_model, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values


def laplace_refinement(
    model: nn.Module,
    train_loader: DataLoader,
    prior_prec: float,
    n_epochs: int = 500,
    optimizer_kwargs: Optional[dict] = None,
    scheduler_cls: LRScheduler = None,
    scheduler_kwargs: Optional[dict] = None,
    eval_frequency: int = 50,
    progress_bar: bool = False,
    sigma_sq: float = 1.0,
    warmstart: bool = False,
) -> Tensor:
    """Update model parameters via the Laplace approximation.

    Assumes model has no buffers (i.e. no batch norm); otherwise will have to
    distinguish these parameters. See "Adapting the Linearised Laplace Model Evidence
    for Modern Deep Learning" [Antoran et. al., 2022]. Solves penalized least-squares
    problem with linearized neural network (convex objective) but without explicitly
    instantiating Jacobians (i.e. using JVPs and VJPs).
        1.  JVP to evaluate linearized neural network predictor in order to evaluate functional
            gradient: f_lin - y
        2.  VJP to evaluate grad of surrogate objective.
    See App. D in [Antoran et. al., 2022] for algorithm outline.
    N.B.: This function has not been adapted for multi-output regression; check linear model
    surrogate gradient evaluation.

    Parameters
    ----------
    model: nn.Module
        The model to be trained.
    train_loader: DataLoader
        Data loader containing the training data.
    prior_prec: float
        Initial prior precision.
    n_epochs: int
        How many epochs the model is trained for.
    optimizer_kwargs: dict
        Keyword arguments of the optimizer.
    scheduler_cls: LRScheduler
        Class defining the desired learning rate scheduler.
    scheduler_kwargs: dict
        Keyword arguments of the scheduler.
    eval_frequency: int
        Define the frequency at which we evaluate the model on the validation data.
        We run the evaluation every `eval_frequency` epochs.
    progress_bar: bool
        Whether to print a progress bar to the screen.
    sigma_sq : float
        Observation noise fixed upfront (assumed previously tuned).
    warmstart : bool
        Whether to initialize linear model parameters at NN parameters or zero vector.

    Returns
    -------
    theta_refine: Tensor
        The estimated observation noise.
    """
    model.eval()
    # Collect all model parameters in a single vector.
    theta_star = parameters_to_vector(model.parameters()).detach()
    device = parameters_to_vector(model.parameters()).device
    if warmstart:
        # Start from the parameters of the linear model.
        theta_refine = nn.Parameter(theta_star.clone()).to(device)
    else:
        # Start from zeros.
        theta_refine = nn.Parameter(torch.zeros_like(theta_star)).to(device)
    # Get function defined by the model, as well as its (non-flat) parameters.
    func_model, params = make_functional(model, disable_autograd_tracking=True)
    theta_diff_unflat = copy.deepcopy(params)
    # Ensure weight decay is set to zero.
    if optimizer_kwargs is not None and "weight_decay" in optimizer_kwargs:
        warnings.warn("Weight decay is handled and optimized. Will be set to 0.")
        optimizer_kwargs["weight_decay"] = 0.0
    # Get data set size and optimization criterion.
    num_train = len(train_loader.dataset)
    criterion = MSELoss(reduction="mean")
    # Set up model optimizer.
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = Adam([theta_refine], **optimizer_kwargs)
    # Set up learning rate scheduler.
    scheduler = None
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    # Initialize list to store the training loss.
    losses = []
    pbar = tqdm.trange(
        1,
        n_epochs + 1,
        disable=not progress_bar,
        position=1,
        leave=False,
        desc="[Training]",
        colour="blue",
    )
    # Expand prior_prec to shp (P,) to handle case of layerwise structure.
    prior_prec_expand = expand_prior_precision(prior_prec, model)
    # Run optimization.
    for epoch in pbar:
        epoch_loss = 0
        # Standard NN training per batch.
        for data in train_loader:
            x, y = data
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Do JVP to evaluate linearized predictive and consequently functional gradient.
            with torch.no_grad():
                theta_diff = theta_refine - theta_star
                jvp_model = lambda params: func_model(params, x)
                vector_to_parameters(theta_diff, theta_diff_unflat)
                _, preds_linear = torch.func.jvp(jvp_model, (params,), (theta_diff_unflat,))
                preds_linear = preds_linear + model(x)
                loss = (
                    criterion(preds_linear, y)
                    + ((prior_prec_expand * sigma_sq * theta_refine) @ theta_refine) / num_train
                )
                func_grad = preds_linear - y
            # Now do VJP.
            optimizer.zero_grad()
            model.zero_grad()
            to_grad = model(x) * func_grad  # Shape of (n_batch, 1).
            to_grad.mean().backward()
            # Collect the gradients.
            grads = []
            for param in model.parameters():
                grads.append(param.grad.detach().flatten())
            grads = torch.cat(grads)
            # Apply penalty given by the laplace approximation.
            gradient_with_penalty = (
                grads + (prior_prec_expand * sigma_sq) * theta_refine.detach() / num_train
            )
            theta_refine.grad = gradient_with_penalty
            optimizer.step()
            epoch_loss += loss.cpu().item() * len(y)
        losses.append(epoch_loss / num_train)
        if scheduler is not None:  # assumes scheduler t_max based on total epochs
            scheduler.step()
        if (epoch % eval_frequency) == 0 or epoch == n_epochs:
            print(f"epoch {epoch}: loss={losses[-1]:.3f}")
    return theta_refine.detach()


def pinball_loss(pred: Tensor, target: Tensor, alpha: float, reduction: str = "mean") -> Tensor:
    """Compute the alpha quantile or pinball loss.

    Parameters
    ----------
    pred: Tensor with shape (num_train,1)
        The predictions made by the model.
    target: Tensor with shape (num_train,1)
        Ground truth values of the target variable.
    alpha: float
        The desired miscoverage rate.
    reduction: str
        Reduction to be applied after computing the loss per data point.

    Returns
    -------
    loss: Tensor
        The pinball loss
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    assert 0 < alpha < 1, "alpha must be in (0, 1)"
    err = target - pred
    loss = torch.max((alpha - 1.0) * err, alpha * err)
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def train_quantile_regression_net(
    model: nn.Module,
    train_loader: DataLoader,
    prior_prec: float = 1.0,
    n_epochs: int = 500,
    quantiles: Optional[List[float]] = None,
    optimizer_kwargs: Optional[dict] = None,
    scheduler_cls: Optional[LRScheduler] = None,
    scheduler_kwargs: Optional[dict] = None,
    progress_bar: bool = False,
) -> Tuple[nn.Module, List[float]]:
    """Train a quantile regression model.

    Only supports single output.

    Parameters
    ----------
    model: nn.Module
        The model to be trained.
    train_loader: DataLoader
        Data loader containing the training data.
    prior_prec: float
        Initial prior precision.
    n_epochs: int
        Number of epochs to train the model for.
    quantiles: List[float]
        Quantiles to optimize for.
    optimizer_kwargs: Optional[dict]
        Keyword parameters of the optimizer.
    scheduler_cls: LRScheduler
        Class defining the desired learning rate scheduler.
    scheduler_kwargs: dict
        Keyword arguments of the scheduler.
    scheduler_kwargs
    progress_bar: bool
        Whether to print a progress bar to the screen.

    Returns
    -------
    model: nn.Module
        The trained model.
    losses: List[float]
        The training loss for each epoch.
    """
    # Ensure weight decay is set to zero.
    if optimizer_kwargs is not None and "weight_decay" in optimizer_kwargs:
        warnings.warn("Weight decay is handled and optimized. Will be set to 0.")
        optimizer_kwargs["weight_decay"] = 0.0
    if quantiles is None:
        quantiles = [0.5]
    # Get device, data set size, number of layers, number of parameters.
    device = parameters_to_vector(model.parameters()).device
    num_train = len(train_loader.dataset)
    # Set up model optimizer.
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    params_opt = list(model.parameters())
    optimizer = Adam(params_opt, **optimizer_kwargs)
    # Set up learning rate scheduler.
    scheduler = None
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    losses = []
    pbar = tqdm.trange(
        1,
        n_epochs + 1,
        disable=not progress_bar,
        position=1,
        leave=False,
        desc="[Training]",
        colour="blue",
    )
    # Run optimization.
    for epoch in pbar:
        epoch_loss = 0
        # Standard NN training per batch.
        for data in train_loader:
            x, y = data
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            theta = parameters_to_vector([p for p in model.parameters() if p.requires_grad])
            f = model(x)  # [B,n_sig_lvls*2]
            loss = 0.0
            for ii, quantile in enumerate(quantiles):
                loss += pinball_loss(f[:, ii].unsqueeze(1), y, quantile)
            loss /= len(quantiles)  # take mean over pinball losses for each quantile
            loss += ((prior_prec * theta) @ theta) / num_train
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()
        if scheduler is not None:
            scheduler.step()
        losses.append(epoch_loss / num_train)
        logging.info(f"[epoch={epoch}]: network training. Loss={losses[-1]:.3f}.")
    return model, losses
