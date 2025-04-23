# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/aleximmer/Laplace
# Copyright (c) 2021 Alex Immer, licensed under the MIT License
# License is provided for attribution purposes only, Not a Contribution

"""Implementation of marginal likelihood optimization during training and post-hoc."""

from __future__ import annotations

import logging
import warnings
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Optional, Type

import numpy as np
import torch
import tqdm
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from laplace.curvature import AsdlGGN
from laplace.curvature.curvature import CurvatureInterface
from laplace.utils import (
    HessianStructure,
    Likelihood,
    PriorStructure,
    SubsetOfWeights,
    expand_prior_precision,
    fix_prior_prec_structure,
)
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.types import Number
from torch.utils.data import DataLoader


def marglik_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    likelihood: Likelihood | str = Likelihood.CLASSIFICATION,
    hessian_structure: HessianStructure | str = HessianStructure.KRON,
    backend: Type[CurvatureInterface] = AsdlGGN,
    optimizer_cls: Type[Optimizer] = Adam,
    optimizer_kwargs: dict | None = None,
    scheduler_cls: Type[LRScheduler] | None = None,
    scheduler_kwargs: dict | None = None,
    n_epochs: int = 300,
    lr_hyp: float = 1e-1,
    lr_hyp_min: float = 1e-1,
    prior_structure: PriorStructure | str = PriorStructure.LAYERWISE,
    n_epochs_burnin: int = 0,
    n_hypersteps: int = 10,
    marglik_frequency: int = 1,
    prior_prec_init: float = 1.0,
    sigma_noise_init: float = 1.0,
    temperature: float = 1.0,
    fix_sigma_noise: bool = False,
    progress_bar: bool = False,
    enable_backprop: bool = False,
    dict_key_x: str = "input_ids",
    dict_key_y: str = "labels",
) -> tuple[BaseLaplace, nn.Module, list[Number], list[Number]]:
    """Run marginal-likelihood based training (Algorithm 1 in [1]).

    Optimize model parameters and hyperparameters jointly.
    Model parameters are optimized to minimize negative log joint (train loss)
    while hyperparameters minimize negative log marginal likelihood.

    This method replaces standard neural network training and adds hyperparameter
    optimization to the procedure.

    The settings of standard training can be controlled by passing `train_loader`,
    `optimizer_cls`, `optimizer_kwargs`, `scheduler_cls`, `scheduler_kwargs`, and `n_epochs`.
    The `model` should return logits, i.e., no softmax should be applied.
    With `likelihood=Likelihood.CLASSIFICATION` or `Likelihood.REGRESSION`, one can choose between
    categorical likelihood (CrossEntropyLoss) and Gaussian likelihood (MSELoss).

    As in [1], we optimize prior precision and, for regression, observation noise
    using the marginal likelihood. The prior precision structure can be chosen
    as `'scalar'`, `'layerwise'`, or `'diagonal'`. `'layerwise'` is a good default
    and available to all Laplace approximations. `lr_hyp` is the step size of the
    Adam hyperparameter optimizer, `n_hypersteps` controls the number of steps
    for each estimated marginal likelihood, `n_epochs_burnin` controls how many
    epochs to skip marginal likelihood estimation, `marglik_frequency` controls
    how often to estimate the marginal likelihood (default of 1 re-estimates
    after every epoch, 5 would estimate every 5-th epoch).

    References
    ----------
    [1] Immer, A., Bauer, M., Fortuin, V., Rätsch, G., Khan, EM.
    Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning ICML 2021
    https://arxiv.org/abs/2104.04975

    Parameters
    ----------
    model : torch.nn.Module
        torch neural network model (needs to comply with Backend choice).
    train_loader : DataLoader
        pytorch dataloader that implements `len(train_loader.dataset)` to obtain number of data
        points.
    likelihood : str
        Likelihood.CLASSIFICATION or Likelihood.REGRESSION.
    hessian_structure : {'diag', 'kron', 'full'}
        structure of the Hessian approximation.
    backend : Backend
        Curvature subclass, e.g. AsdlGGN/AsdlEF or BackPackGGN/BackPackEF.
    optimizer_cls : torch.optim.Optimizer
        optimizer to use for optimizing the neural network parameters togeth with `train_loader`.
    optimizer_kwargs : dict
        keyword arguments for `optimizer_cls`, for example to change learning rate or momentum.
    scheduler_cls : torch.optim.lr_scheduler._LRScheduler
        optionally, a scheduler to use on the learning rate of the optimizer.
        `scheduler.step()` is called after every batch of the standard training.
    scheduler_kwargs : dict
        keyword arguments for `scheduler_cls`, e.g. `lr_min` for CosineAnnealingLR.
    n_epochs : int
        number of epochs to train for.
    lr_hyp : float
        Adam learning rate for hyperparameters.
    lr_hyp_min: float
        Minimum learning rate the scheduler can set to update hyperparameters.
    prior_structure : str
        structure of the prior. one of `['scalar', 'layerwise', 'diag']`.
    n_epochs_burnin : int
        how many epochs to train without estimating and differentiating marglik.
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated.
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood.
        `marglik_frequency=1` would be every epoch,
        `marglik_frequency=5` would be every 5 epochs.
    prior_prec_init : float
        initial prior precision.
    sigma_noise_init : float
        initial observation noise (for regression only).
    temperature : float
        factor for the likelihood for 'overcounting' data. Might be required for data augmentation.
    fix_sigma_noise: bool
        if False, optimize observation noise via marglik otherwise use `sigma_noise_init`.
        throughout. Only works for regression.
    progress_bar: bool
        whether to show a progress bar (updated per epoch) or not.
    enable_backprop : bool
        make the returned Laplace instance backpropable---useful for e.g. Bayesian optimization.
    dict_key_x: str
        The dictionary key under which the input tensor `x` is stored. Only has effect when the
        model takes a `MutableMapping` as the input. Useful for Huggingface LLM models.
    dict_key_y: str
        The dictionary key under which the target tensor `y` is stored. Only has effect when the
        model takes a `MutableMapping` as the input. Useful for Huggingface LLM models.

    Returns
    -------
    lap: BaseLaplace
        Fit Laplace approximation with the best obtained marginal likelihood during training.
    model: nn.Module
        Corresponding model with the MAP parameters.
    margliks: List[Number]
        List of marginal likelihoods obtained during training (to monitor convergence).
    losses: List[Number]
        List of losses (log joints) obtained during training (to monitor convergence).
    """
    if optimizer_kwargs is not None and "weight_decay" in optimizer_kwargs:
        warnings.warn("Weight decay is handled and optimized. Will be set to 0.")
        optimizer_kwargs["weight_decay"] = 0.0

    device = parameters_to_vector(model.parameters()).device
    num_train = len(train_loader.dataset)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_layers = len(trainable_params)
    num_params = len(parameters_to_vector(trainable_params))

    # Differentiable hyperparameters.
    hyperparameters = []
    # Set prior precision.
    log_prior_prec_init = np.log(temperature * prior_prec_init)
    log_prior_prec = fix_prior_prec_structure(
        log_prior_prec_init, prior_structure, num_layers, num_params, device
    )
    log_prior_prec.requires_grad = True
    hyperparameters.append(log_prior_prec)

    # Set up loss (and observation noise hyperparam).
    if likelihood == Likelihood.CLASSIFICATION:
        criterion = CrossEntropyLoss(reduction="mean")
        sigma_noise = 1.0
    elif likelihood == Likelihood.REGRESSION:
        criterion = MSELoss(reduction="mean")
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # Set up model optimizer.
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # Set up learning rate scheduler.
    scheduler = None
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    # Set up hyperparameter optimizer.
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)

    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    best_sigma = 1
    losses = []
    margliks = []

    pbar = tqdm.trange(
        1,
        n_epochs + 1,
        disable=not progress_bar,
        position=1,
        leave=False,
        desc="[Training]",
        colour="blue",
    )

    for epoch in pbar:
        epoch_loss = 0
        epoch_perf = 0

        # Standard NN training per batch.
        for data in train_loader:
            if isinstance(data, MutableMapping):
                x, y = data, data[dict_key_y]
                y = y.to(device, non_blocking=True)
            else:
                x, y = data
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if likelihood == Likelihood.REGRESSION:
                sigma_noise = (
                    torch.exp(log_sigma_noise).detach() if not fix_sigma_noise else sigma_noise_init
                )
                crit_factor = temperature / (2 * sigma_noise**2)
                K = y.shape[1]  # for regression also rescale weight decay by n_outputs
            else:
                crit_factor = temperature
                K = 1

            prior_prec = torch.exp(log_prior_prec).detach()
            theta = parameters_to_vector([p for p in model.parameters() if p.requires_grad])
            delta = expand_prior_precision(prior_prec, model)

            f = model(x)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / (num_train * K) / crit_factor
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() * len(y)

            if likelihood == Likelihood.REGRESSION:
                epoch_perf += (f.detach() - y).square().sum()
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item()

        # There was a bug here in the original marglik code
        # The scheduler should be called only every epoch.
        if scheduler is not None:
            scheduler.step()

        losses.append(epoch_loss / num_train)

        # Compute validation error to report during training.
        logging.info(
            f"MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]:.3f}."
            + f"Perf={epoch_perf/num_train:.3f}"
        )

        # Only update hyperparameters every marglik_frequency steps after burnin
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # Optimizer hyperparameters by differentiating marglik.
        # 1. Fit laplace approximation
        if likelihood == Likelihood.CLASSIFICATION:
            sigma_noise = 1
        else:
            sigma_noise = torch.exp(log_sigma_noise) if not fix_sigma_noise else sigma_noise_init
        prior_prec = torch.exp(log_prior_prec)
        lap = Laplace(
            model,
            likelihood,
            hessian_structure=hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_prec,
            temperature=temperature,
            backend=backend,
            subset_of_weights="all",
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
        )
        lap.fit(train_loader)

        if likelihood == Likelihood.REGRESSION:
            K = lap.n_outputs
        else:
            K = 1

        # 2. Differentiate wrt. hyperparameters for n_hypersteps.
        for _ in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            if likelihood == Likelihood.CLASSIFICATION or fix_sigma_noise:
                sigma_noise = None
            else:
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / (num_train * K)
            marglik.backward()
            hyper_optimizer.step()
            hyper_scheduler.step()
            margliks.append(marglik.item())

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_precision = deepcopy(prior_prec.detach())
            if likelihood != Likelihood.CLASSIFICATION:
                best_sigma = (
                    deepcopy(sigma_noise.detach()) if not fix_sigma_noise else sigma_noise_init
                )
            best_marglik = margliks[-1]
            logging.info(
                f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}. "
                + "Saving new best model."
            )
        else:
            logging.info(
                f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}."
                + f"No improvement over {best_marglik:.2f}"
            )

    logging.info("MARGLIK: finished training. Recover best model and fit Laplace.")

    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    logging.info(f"best params: {sigma_noise}, {prior_prec}")

    lap = Laplace(
        model,
        likelihood,
        hessian_structure=hessian_structure,
        sigma_noise=sigma_noise,
        prior_precision=prior_prec,
        temperature=temperature,
        backend=backend,
        subset_of_weights=SubsetOfWeights.ALL,
        enable_backprop=enable_backprop,
        dict_key_x=dict_key_x,
        dict_key_y=dict_key_y,
    )
    lap.fit(train_loader)
    return lap, model, margliks, losses


def marglik_training_posthoc(
    model: torch.nn.Module,
    train_loader: DataLoader,
    likelihood: Likelihood | str = Likelihood.CLASSIFICATION,
    hessian_structure: HessianStructure | str = HessianStructure.KRON,
    backend: Type[CurvatureInterface] = AsdlGGN,
    n_steps: int = 100,
    lr_hyp: float = 1e-1,
    lr_hyp_min: float = 1e-1,
    prior_prec_init: float = 1.0,
    sigma_noise_init: float = 1.0,
    prior_structure: PriorStructure | str = PriorStructure.LAYERWISE,
    temperature: float = 1.0,
    fix_sigma_noise: bool = False,
    subset_of_weights: SubsetOfWeights | str = SubsetOfWeights.ALL,
    laplace_kwargs: Optional[dict] = None,
) -> tuple[BaseLaplace, nn.Module, list[Number]]:
    """Run marginal likelihood optimization on existing pretrained model.

    Tune sigma_noise, prior_prec post-hoc on the data provided in `train_loader`.
    The model is assumed to be pretrained and remains unchanged, hence why the optimization
    is deemed posthoc.

    Parameters
    ----------
    model: nn.Module
        Torch neural network model.
    train_loader: DataLoader
        Data loader containing the training data.
    likelihood : str
        Likelihood.CLASSIFICATION or Likelihood.REGRESSION.
    hessian_structure : {'diag', 'kron', 'full'}
        structure of the Hessian approximation.
    backend : Backend
        Curvature subclass, e.g. AsdlGGN/AsdlEF or BackPackGGN/BackPackEF.
    n_steps: int
        Number of optimization steps.
    lr_hyp: float
        Learning rate used to update hyperparameters.
    lr_hyp_min: float
        Minimum learning rate the scheduler can set to update hyperparameters.
    prior_prec_init: float
        Initial prior precision.
    sigma_noise_init: float
        Initial observation noise.
    prior_structure : str
        structure of the prior. one of `['scalar', 'layerwise', 'diag']`.
    temperature : float
        factor for the likelihood for 'overcounting' data. Might be required for data augmentation.
    fix_sigma_noise: bool
        if False, optimize observation noise via marglik otherwise use `sigma_noise_init`.
        throughout. Only works for regression.
    subset_of_weights: str
        Defines which weights of the network will be used to compute the Laplace posterior
        approximation. Can be either one of {'last_layer', 'subnetwork', 'all'}.
    laplace_kwargs: dict
        Keyword arguments to the Laplace object.

    Returns
    -------
    lap: BaseLaplace
        Fit Laplace approximation with the best obtained marginal likelihood during training.
    model: nn.Module
        Torch neural network model.
    margliks: List[Number]
        The marginal likelihood of each optimization step.
    """
    if laplace_kwargs is None:
        laplace_kwargs = {}
    # Get device, data set size, number of layers, number of parameters.
    device = parameters_to_vector(model.parameters()).device
    num_train = len(train_loader.dataset)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_layers = len(trainable_params)
    num_params = len(parameters_to_vector(trainable_params))
    # Differentiable hyperparameters.
    hyperparameters = []
    # Set prior precision.
    log_prior_prec_init = np.log(temperature * prior_prec_init)
    log_prior_prec = fix_prior_prec_structure(
        log_prior_prec_init, prior_structure, num_layers, num_params, device
    )
    log_prior_prec.requires_grad = True
    hyperparameters.append(log_prior_prec)
    # Set up loss (and observation noise hyperparam).
    if likelihood == Likelihood.CLASSIFICATION:
        sigma_noise = 1.0
    elif likelihood == Likelihood.REGRESSION:
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)
    if likelihood == Likelihood.CLASSIFICATION:
        sigma_noise = 1
    else:
        sigma_noise = torch.exp(log_sigma_noise) if not fix_sigma_noise else sigma_noise_init
    prior_prec = torch.exp(log_prior_prec)
    # Construct Laplace approximation on the training data.
    lap = Laplace(
        model,
        likelihood,
        hessian_structure=hessian_structure,
        sigma_noise=sigma_noise,
        prior_precision=prior_prec,
        temperature=temperature,
        backend=backend,
        subset_of_weights=subset_of_weights,
        **laplace_kwargs,
    )
    lap.fit(train_loader)

    if likelihood == Likelihood.REGRESSION:
        K = lap.n_outputs
    else:
        K = 1
    # Set up hyperparameter optimizer.
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)
    # Optimize the hyperparameters to maximize the marginal likelihood.
    margliks = []
    for _ in range(n_steps):
        hyper_optimizer.zero_grad()
        if likelihood == Likelihood.CLASSIFICATION or fix_sigma_noise:
            sigma_noise = None
        else:
            sigma_noise = torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / (num_train * K)
        marglik.backward()
        hyper_optimizer.step()
        hyper_scheduler.step()
        margliks.append(marglik.item())
    print(f"marglik optimization. MargLik={margliks[-1]:.2f}.")
    best_precision = deepcopy(prior_prec.detach())
    if likelihood == Likelihood.CLASSIFICATION:
        best_sigma = 1
    else:
        best_sigma = deepcopy(sigma_noise.detach()) if not fix_sigma_noise else sigma_noise_init
    print(f"best params: sigma_noise={sigma_noise}, prior_prec={prior_prec}")
    # Construct Laplace approximation using the best hyperparameters.
    lap = Laplace(
        model,
        likelihood,
        hessian_structure=hessian_structure,
        sigma_noise=best_sigma,
        prior_precision=best_precision,
        temperature=temperature,
        backend=backend,
        subset_of_weights=subset_of_weights,
        **laplace_kwargs,
    )
    lap.fit(train_loader)
    return lap, model, margliks
