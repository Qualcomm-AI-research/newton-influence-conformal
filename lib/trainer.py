# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Training routines for regression models with Laplace approximation."""

from typing import Tuple

import numpy as np
import torch
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from laplace.curvature.asdl import AsdlGGN
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lib.linalg_utils import cholesky_add_jitter_until_psd
from lib.marglik_training import marglik_training, marglik_training_posthoc
from lib.nn_training import train_regression_net
from lib.utils import reset_parameters


def train_laplace_model(
    hyperparam_opt: str,
    model: nn.Module,
    train_loader_full: DataLoader,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    n_epochs: int,
    n_epochs_cv: int,
    lr: float,
    lr_cv: float,
    lr_min: float = 1e-5,
    lr_hyp: float = 1e-2,
    lr_hyp_min: float = 1e-3,
    n_epochs_burnin: int = 100,
    n_hypersteps: int = 50,
    marglik_frequency: int = 50,
    prior_prec_init: float = 1.0,
    prior_prec_log_lower: float = -4.0,
    prior_prec_log_upper: float = 4.0,
    hessian_structure: str = "full",
    subset_of_weights: str = "all",
) -> Tuple[BaseLaplace, BaseLaplace]:
    """Train regression model and compute Laplace approximation.

    Hyperparameters are fit either via cross- validation or
    marginal likelihood optimization. Return laplace posterior approximation based on
    the learned model.

    Parameters
    ----------
    hyperparam_opt: str
        Determines which type of hyperparameter optimization to use. Either 'marglik' or 'cv'.
    model: nn.Module
        The model to be trained.
    train_loader_full: Dataloader
        Data loader containing all the training data.
    train_loader: Dataloader
        Data loader containing all the training data except for the data reserved for validation.
    valid_loader: Dataloader
        Data loader containing the validation data.
    n_epochs: int
        Number of epochs to train the model for when optimizing the marginal likelihood.
    n_epochs_cv: int
        Number of epochs to train the model for when fitting hyperparameters via cross-validation.
    lr: float
        Learning rate used to train main model when optimizing the marginal likelihood.
    lr_cv: float
        Learning rate used to train the model when fitting hyperparameters via cross-validation.
    lr_min: float
        Minimum learning rate the scheduler can set.
    lr_hyp: float
        Learning rate used to update hyperparameters.
    lr_hyp_min: float
        Minimum learning rate the scheduler can set to update hyperparameters.
    n_epochs_burnin: int
        Number of epochs to train before starting to estimate the marginal likelihood.
    n_hypersteps: int
        How many steps to take on the hyperparameters when marginal likelihood is estimated.
    marglik_frequency
        How often to estimate (and differentiate) the marginal likelihood.
        `marglik_frequency=1` would be every epoch,
        `marglik_frequency=5` would be every 5 epochs.
    prior_prec_init: float
        Initial prior precision.
    prior_prec_log_lower: float
        Lowest value of log prior precision to consider.
    prior_prec_log_upper: float
        Highest value of log prior precision to consider.
    hessian_structure: str
        Structure of the Hessian approximation.
        Can be either one of {'diag', 'kron', 'full'}, default='kron'.
    subset_of_weights: str
        Defines which weights of the network will be used to compute the Laplace posterior
        approximation. Can be either one of {'last_layer', 'subnetwork', 'all'}.

    Returns
    -------
    la: BaseLaplace
        Laplace approximation with the best model fit via marginal likelihood or cross-validation.
    la_bayes: BaseLaplace
        Laplace approximation with the best model fit via marginal likelihood or cross-validation
        computed with the ML estimate of observation noise. This is used to compute confidence
        intervals using the approximate posterior.
    """
    if hyperparam_opt == "marglik":
        # Joint MAP training and hyperparameter (L2-reg, obs noise) tuning via marginal likelihood.
        la, _, _, _ = marglik_training(
            model=model,
            train_loader=train_loader_full,
            likelihood="regression",
            hessian_structure=hessian_structure,
            backend=AsdlGGN,
            prior_structure="layerwise",
            n_epochs=n_epochs,
            optimizer_cls=Adam,
            optimizer_kwargs={"lr": lr},
            scheduler_cls=CosineAnnealingLR,
            scheduler_kwargs={"T_max": n_epochs, "eta_min": lr_min},
            lr_hyp=lr_hyp,
            lr_hyp_min=lr_hyp_min,
            n_epochs_burnin=n_epochs_burnin,
            n_hypersteps=n_hypersteps,
            marglik_frequency=marglik_frequency,
            prior_prec_init=prior_prec_init,
            sigma_noise_init=1.0,
            temperature=1.0,
            fix_sigma_noise=False,
        )
        if hessian_structure == "full":
            la._posterior_scale = cholesky_add_jitter_until_psd(la.posterior_precision)
            torch.cuda.empty_cache()

        la_bayes = la
    elif hyperparam_opt == "cv":
        prior_precs = 10.0 ** np.arange(
            prior_prec_log_lower, prior_prec_log_upper + 1, dtype=np.float32
        )
        train_map_kwargs = {"optimizer_kwargs": {"lr": lr_cv}, "scheduler_cls": CosineAnnealingLR}
        # Perform grid search over prior_prec values.
        nlls = []
        for prior_prec in prior_precs:
            reset_parameters(model)
            # NB: valid_nll is evaluated with sigma_noise=1 when sigma_noise_posthoc=True (default).
            _, _, _, valid_nlls = train_regression_net(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                prior_prec=prior_prec,
                n_epochs=n_epochs_cv,
                scheduler_kwargs={"T_max": n_epochs_cv, "eta_min": lr_min},
                **train_map_kwargs,
            )
            print(f"prior_prec {prior_prec:.1e}  val-nll {valid_nlls[-1]:.5f}")
            nlls.append(valid_nlls[-1])

        # Retrain on full train set given optimal prior_prec.
        opt_prior_prec = prior_precs[np.argmin(nlls)]
        print(f"optimal prior_prec {opt_prior_prec:.1e}")
        reset_parameters(model)
        model, sigma_noise, _, _ = train_regression_net(
            model=model,
            train_loader=train_loader_full,
            prior_prec=opt_prior_prec,
            n_epochs=n_epochs,
            scheduler_kwargs={"T_max": n_epochs, "eta_min": lr_min},
            **train_map_kwargs,
        )

        lap_kwargs = {
            "likelihood": "regression",
            "hessian_structure": hessian_structure,
            "prior_precision": opt_prior_prec,
            "temperature": 1.0,
            "backend": AsdlGGN,
            "subset_of_weights": subset_of_weights,
        }

        # Construct laplace object just for evaluating bayes intervals.
        # Use MLE-fitted sigma_noise in Hessian/covariance computation.
        la_bayes = Laplace(model=model, sigma_noise=sigma_noise, **lap_kwargs)
        la_bayes.fit(train_loader_full)
        if hessian_structure == "full":
            la_bayes._posterior_scale = cholesky_add_jitter_until_psd(la_bayes.posterior_precision)
            torch.cuda.empty_cache()

        # NB: unnecessary repeated computation -- Hessian computation for la_bayes almost same.
        # Only using la_bayes to evaluate var_test so could hardcode computation.
        la = Laplace(model=model, sigma_noise=1.0, **lap_kwargs)
        la.fit(train_loader_full)
        if hessian_structure == "full":
            la._posterior_scale = cholesky_add_jitter_until_psd(la.posterior_precision)
            torch.cuda.empty_cache()
    else:
        raise ValueError("Unrecognised hyperparam_opt")

    return la, la_bayes


def train_laplace_model_calib_split(
    hyperparam_opt: str,
    model: nn.Module,
    train_loader_full: DataLoader,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    calib_loader: DataLoader,
    n_epochs: int,
    n_epochs_cv: int,
    lr: float,
    lr_cv: float,
    lr_min: float = 1e-5,
    lr_hyp: float = 1e-2,
    lr_hyp_min: float = 1e-3,
    n_epochs_burnin: int = 100,
    n_hypersteps: int = 50,
    marglik_frequency: int = 50,
    prior_prec_init: float = 1.0,
    prior_prec_log_lower: float = -4.0,
    prior_prec_log_upper: float = 4.0,
    hessian_structure="full",
    subset_of_weights="all",
) -> Tuple[BaseLaplace, BaseLaplace]:
    """Train regression model and get Laplace approximation only on calibration data.

    Hyperparameters are fit either via cross- validation or
    marginal likelihood optimization on the entire training data. The Laplace
    approximation plus linearization is then applied to the calibration data to estimate
    the predictive variance of each data point. Return laplace posterior approximation
    based on the learned model.

    Parameters
    ----------
    hyperparam_opt: str
        Determines which type of hyperparameter optimization to use. Either 'marglik' or 'cv'.
    model: nn.Module
        The model to be trained.
    train_loader_full: Dataloader
        Data loader containing all the training data.
    train_loader: Dataloader
        Data loader containing all the training data except for the data reserved for validation.
    valid_loader: Dataloader
        Data loader containing the validation data.
    calib_loader: Dataloader
        Data loader containing the calibration data.
    n_epochs: int
        Number of epochs to train the model for when optimizing the marginal likelihood.
    n_epochs_cv: int
        Number of epochs to train the model for when fitting hyperparameters via cross-validation.
    lr: float
        Learning rate used to train main model when optimizing the marginal likelihood.
    lr_cv: float
        Learning rate used to train the model when fitting hyperparameters via cross-validation.
    lr_min: float
        Minimum learning rate the scheduler can set.
    lr_hyp: float
        Learning rate used to update hyperparameters.
    lr_hyp_min: float
        Minimum learning rate the scheduler can set to update hyperparameters.
    n_epochs_burnin: int
        Number of epochs to train before starting to estimate the marginal likelihood.
    n_hypersteps: int
        How many steps to take on the hyperparameters when marginal likelihood is estimated.
    marglik_frequency
        How often to estimate (and differentiate) the marginal likelihood.
        `marglik_frequency=1` would be every epoch,
        `marglik_frequency=5` would be every 5 epochs.
    prior_prec_init: float
        Initial prior precision.
    prior_prec_log_lower: float
        Lowest value of log prior precision to consider.
    prior_prec_log_upper: float
        Highest value of log prior precision to consider.
    hessian_structure: str
        Structure of the Hessian approximation.
        Can be either one of {'diag', 'kron', 'full'}, default='kron'.
    subset_of_weights: str
        Defines which weights of the network will be used to compute the Laplace posterior
        approximation. Can be either one of {'last_layer', 'subnetwork', 'all'}.

    Returns
    -------
    la: BaseLaplace
        Laplace approximation with the best model fit via marginal likelihood or cross-validation.
    la_bayes: BaseLaplace
        Laplace approximation with the best model fit via marginal likelihood or cross-validation
        computed with the ML estimate of observation noise. This is used to compute confidence
        intervals using the approximate posterior.
    """
    if hyperparam_opt == "marglik":
        # Do usual MAP training first.
        prior_prec = 1e-4 * len(train_loader_full.dataset)
        print(f"optimal prior_prec {prior_prec:.1e}")
        model, _, _, _ = train_regression_net(
            model=model,
            train_loader=train_loader_full,
            prior_prec=prior_prec,
            n_epochs=n_epochs,
            scheduler_kwargs={"T_max": n_epochs, "eta_min": lr_min},
            optimizer_kwargs={"lr": lr_cv},
            scheduler_cls=CosineAnnealingLR,
        )

        # Treat model above as (static) pre-trained network.
        # Then, tune sigma_noise, prior_prec post-hoc on calibration set.
        la, model, _ = marglik_training_posthoc(
            model=model,
            train_loader=calib_loader,
            likelihood="regression",
            hessian_structure=hessian_structure,
            backend=AsdlGGN,
            prior_structure="layerwise",
            n_steps=n_epochs,
            lr_hyp=lr_hyp,
            lr_hyp_min=lr_hyp_min,
            prior_prec_init=prior_prec_init,
            sigma_noise_init=1.0,
            temperature=1.0,
            fix_sigma_noise=False,
        )

        if hessian_structure == "full":
            la._posterior_scale = cholesky_add_jitter_until_psd(la.posterior_precision)
            torch.cuda.empty_cache()

        la_bayes = la
    elif hyperparam_opt == "cv":
        prior_precs = 10.0 ** np.arange(
            prior_prec_log_lower, prior_prec_log_upper + 1, dtype=np.float32
        )
        train_map_kwargs = {"optimizer_kwargs": {"lr": lr_cv}, "scheduler_cls": CosineAnnealingLR}

        opt_prior_prec = 1e-4 * len(train_loader_full.dataset)
        print(f"optimal prior_prec {opt_prior_prec:.1e}")
        reset_parameters(model)
        model, _, _, _ = train_regression_net(
            model=model,
            train_loader=train_loader_full,
            prior_prec=opt_prior_prec,
            n_epochs=n_epochs,
            scheduler_kwargs={"T_max": n_epochs, "eta_min": lr_min},
            **train_map_kwargs,
        )

        # Fit sigma_noise by MLE on calibration set.
        N_calib = len(calib_loader.dataset)
        ssqe = 0
        for x, y in calib_loader:
            with torch.no_grad():
                ssqe += (y - model(x)).square().sum().item() / N_calib
        sigma_noise = np.sqrt(np.clip(ssqe, 1e-4, None)).item()

        lap_kwargs = {
            "likelihood": "regression",
            "hessian_structure": hessian_structure,
            "prior_precision": opt_prior_prec,
            "temperature": 1.0,
            "backend": AsdlGGN,
            "subset_of_weights": subset_of_weights,
        }

        # Construct laplace object just for evaluating bayes intervals.
        # Use MLE-fitted sigma_noise in Hessian/covariance computation.
        la_bayes = Laplace(model=model, sigma_noise=sigma_noise, **lap_kwargs)
        la_bayes.fit(calib_loader)

        print("Tuning prior_prec for la_bayes")
        margliks = []
        for prior_prec in prior_precs:
            marglik = la_bayes.log_marginal_likelihood(torch.tensor(prior_prec))
            print(f"prior_prec {prior_prec:.1e}  marglik {marglik:.5f}")
            margliks.append(marglik.item())
        opt_prior_prec = prior_precs[np.nanargmax(margliks)]
        la_bayes.prior_precision = torch.tensor(opt_prior_prec)

        if hessian_structure == "full":
            la_bayes._posterior_scale = cholesky_add_jitter_until_psd(la_bayes.posterior_precision)
            torch.cuda.empty_cache()

        # NB: unnecessary repeated computation -- Hessian computation for la_bayes almost same.
        # Only using la_bayes to evaluate fvar_test so could hardcode computation.
        la = Laplace(model=model, sigma_noise=1.0, **lap_kwargs)
        la.fit(calib_loader)

        print("Tuning prior_prec for la (CRR)")
        margliks = []
        for prior_prec in prior_precs:
            marglik = la.log_marginal_likelihood(torch.tensor(prior_prec))
            print(f"prior_prec {prior_prec:.1e}  marglik {marglik:.5f}")
            margliks.append(marglik.item())
        opt_prior_prec = prior_precs[np.nanargmax(margliks)]
        la.prior_precision = torch.tensor(opt_prior_prec)

        if hessian_structure == "full":
            la._posterior_scale = cholesky_add_jitter_until_psd(la.posterior_precision)
            torch.cuda.empty_cache()
    else:
        raise ValueError("Unrecognised hyperparam_opt")

    return la, la_bayes
