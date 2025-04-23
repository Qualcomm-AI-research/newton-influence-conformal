# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Script to run UCI regression experiment."""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data as data_utils
from laplace import Laplace, marglik_training
from laplace.baselaplace import BaseLaplace
from laplace.curvature.asdl import AsdlGGN
from numpy.typing import NDArray
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.crr import CRR_ALGOS, eval_bayes, eval_crr, eval_split_cp
from lib.nn_training import train_regression_net
from lib.utils import reset_parameters, set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_synthetic_data(
    with_outliers: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate synthetic data experiment from Sec.

    5.1 / Table 1+2 in [Papadopoulos, 2023]
    "Guaranteed Coverage Prediction Intervals with Gaussian Process Regression".
    Data generated from GP prior with RBF kernel, w/o outliers.
    """
    n_train = 500
    n_test = 100
    x_all = torch.randn(n_train + n_test, device=DEVICE)  # (N+M,)
    # Generate outlier observations given by higher observation noise.
    p_outlier = 0.1 if with_outliers else 0.0
    mask_outlier = torch.bernoulli(p_outlier * torch.ones(n_train + n_test, device=DEVICE)).bool()
    obs_noise = 0.01 * torch.ones(n_train + n_test, device=DEVICE)
    obs_noise[mask_outlier] = 1.0
    # Evaluate SE kernel (with unit lengthscale and unit scale) on inputs corrupted by Gaussian
    # noise.
    dist = x_all.unsqueeze(1) - x_all.unsqueeze(0)
    cov = torch.exp(-0.5 * dist.pow(2)) + torch.diag_embed(obs_noise)  # (N+M,N+M)
    y_all = MultivariateNormal(torch.zeros(n_train + n_test, device=DEVICE), cov).rsample()
    x_train = x_all[:n_train].unsqueeze(-1)
    x_test = x_all[n_train:].unsqueeze(-1)
    y_train = y_all[:n_train].unsqueeze(-1)
    y_test = y_all[n_train:].unsqueeze(-1)
    return x_train, y_train, x_test, y_test


def train_map(
    model: nn.Module,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
) -> nn.Module:
    """Train the model (maximum as posteriori) on the given data.

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.

    Returns
    -------
    model: nn.Module
        The trained model.
    """
    ds_train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(ds_train, batch_size=len(x_train), shuffle=True)
    n_epochs = 500
    train_map_kwargs = {
        "n_epochs": n_epochs,
        "prior_prec": opt_prior_prec * opt_sigma_noise**2,
        "optimizer_kwargs": {"lr": 1e-2},
        "scheduler_cls": CosineAnnealingLR,
        "scheduler_kwargs": {"T_max": n_epochs, "eta_min": 1e-5},
    }
    model.train()
    model, _, _, _ = train_regression_net(
        model=model, train_loader=train_loader, **train_map_kwargs
    )
    model.eval()
    with torch.no_grad():
        preds_test = model(x_test)
        test_mse = (preds_test.squeeze() - y_test.squeeze()).square().sum().item() / len(y_test)
        test_nll = -Normal(loc=preds_test, scale=1.0).log_prob(y_test).sum() / len(y_test)
    print(f"test_mse {test_mse}  test_nll {test_nll}")
    return model


def train_and_lap(
    model: nn.Module,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
) -> BaseLaplace:
    """Train model and Laplace approximation on the given data.

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.

    Returns
    -------
    la: BaseLaplace
        Laplace approximation with the best model fit via marginal likelihood or cross-validation.
    """
    model = train_map(model, opt_prior_prec, opt_sigma_noise, x_train, y_train, x_test, y_test)
    ds_train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(ds_train, batch_size=len(x_train))
    la = Laplace(
        model=model,
        sigma_noise=opt_sigma_noise,
        likelihood="regression",
        hessian_structure="full",
        prior_precision=opt_prior_prec,
        temperature=1.0,
        backend=AsdlGGN,
        subset_of_weights="all",
    )
    la.fit(train_loader)
    return la


def run_bayes(
    model: nn.Module,
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    sig_lvls: List[float],
) -> NDArray:
    """Compute confidence intervals via Bayes posterior predictive distribution.

    Train the model and use the Laplace approximation to compute confidence intervals
    using the approximate posterior predictive distribution.

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.
    sig_lvls: List[float]
        List of significance levels to construct PI.

    Returns
    -------
    cp_intervals: NDArray
        The intervals construct by the posterior predictive distribution.
    """
    la = train_and_lap(model, opt_prior_prec, opt_sigma_noise, x_train, y_train, x_test, y_test)
    jacobian_test, preds_test = la.backend.jacobians(x_test)
    h_test = la.functional_variance(jacobian_test)
    cp_intervals = eval_bayes(
        fmu=preds_test.squeeze().cpu().detach().numpy(),
        fvar=h_test.squeeze().cpu().detach().numpy(),
        sigma_sq=la.sigma_noise.item() ** 2,
        sig_lvls=sig_lvls,
    )
    return cp_intervals


def run_acpgn(
    model: nn.Module,
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    sig_lvls: List[float],
    crr_algo: str,
) -> NDArray:
    """Run ACP-GN (approximate full conformal prediction with Gauss-Newton influence).

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.
    sig_lvls: List[float]
        List of significance levels to construct PI.
    crr_algo: str
        CRR implementation choice of "nouretdinov", "vovk", "vovk_mod", "burnaev".

    Returns
    -------
    cp_intervals: NDArray
        The intervals construct by ACP-GN.
    """
    la = train_and_lap(model, opt_prior_prec, opt_sigma_noise, x_train, y_train, x_test, y_test)

    # predictions on train data
    jacobian, preds = la.backend.jacobians(x_train)
    # "leverage" on train points (only needed for jackknife residual)
    h_train = la.functional_variance(jacobian)
    # reshape Jacobian needed for cross-leverage computation
    n_batch, n_outs, n_params = jacobian.shape
    jacobian = jacobian.reshape(n_batch * n_outs, n_params)

    # "leverage" and predictions on test points
    jacobian_test, preds_test = la.backend.jacobians(x_test)
    h_test = la.functional_variance(jacobian_test)
    # "cross-leverage" between test points and train data
    jacobian_test = jacobian_test.reshape(len(x_test) * n_outs, n_params)
    h_test_train = torch.einsum("mp,pq,nq->mn", jacobian_test, la.posterior_covariance, jacobian)
    intvl_kwargs = {
        "ys": y_train.squeeze().cpu().detach().numpy(),
        "preds": preds.squeeze().cpu().detach().numpy(),
        "preds_eval": preds_test.squeeze().cpu().detach().numpy(),
        "h_mn": h_test_train.squeeze().cpu().detach().numpy(),
        "h_n": h_train.squeeze().cpu().detach().numpy(),
        "sigma_sq": la.sigma_noise.item() ** 2,
        "sig_lvls": sig_lvls,
    }
    cp_intervals = eval_crr(
        h_m=h_test.squeeze().cpu().detach().numpy(),
        algo=crr_algo,
        nonconformity_score="standard",
        **intvl_kwargs,
    )
    return cp_intervals


def run_scp(
    model: nn.Module,
    x_train: NDArray,
    y_train: NDArray,
    x_calib: NDArray,
    y_calib: NDArray,
    x_test: NDArray,
    y_test: NDArray,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    sig_lvls: List[float],
) -> NDArray:
    """Train the model and runs split conformal prediction on the given data.

    A separation data split is required for calibration.

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_calib: NDArray
        Input data used for calibration.
    y_calib: NDArray
        Labels for the calibration data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.
    sig_lvls: List[float]
        List of significance levels to construct PI.

    Returns
    -------
    cp_intervals: NDArray
        The intervals construct by Split CP.
    """
    model = train_map(model, opt_prior_prec, opt_sigma_noise, x_train, y_train, x_test, y_test)
    intvl_kwargs = {
        "ys_calib": y_calib.squeeze().cpu().detach().numpy(),
        "preds_calib": model(x_calib).squeeze().cpu().detach().numpy(),
        "preds_eval": model(x_test).squeeze().cpu().detach().numpy(),
        "sigma_sq": opt_sigma_noise**2,
        "sig_lvls": sig_lvls,
    }
    cp_intervals = eval_split_cp(**intvl_kwargs)
    return cp_intervals


def run_fullcp(
    model: nn.Module,
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    sig_lvls: List[float],
    n_grid: int,
) -> NDArray:
    """Run full conformal prediction, that is, retraining the model each time.

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.
    sig_lvls: List[float]
        List of significance levels to construct PI.
    n_grid: int
        Number of points in the grid used for postulated labels.

    Returns
    -------
    cp_intervals: NDArray
        The intervals construct by Full CP.
    """
    sig_lvls = np.array(sig_lvls)
    y_grid = torch.linspace(y_train.min(), y_train.max(), n_grid, device=DEVICE)
    n_train = len(x_train)
    n_test = len(x_test)
    conf_threshold = np.ceil((n_train + 1) * (1 - sig_lvls)).astype(int)
    cp_intervals = []
    for idx_test in range(n_test):
        cp_set_all = [[] for _ in range(len(sig_lvls))]
        for idx_grid in range(n_grid):
            reset_parameters(model)
            print(f"#test point {idx_test+1}/{n_test}  #grid point {idx_grid+1}/{n_grid}")
            # (x_N+1, y) appended to end of train set
            x_train_aug = torch.cat((x_train, x_test[idx_test].unsqueeze(1)))
            y_train_aug = torch.cat((y_train, y_grid[idx_grid][None].unsqueeze(1)))
            model = train_map(
                model, opt_prior_prec, opt_sigma_noise, x_train_aug, y_train_aug, x_test, y_test
            )
            model.eval()
            with torch.no_grad():
                preds_aug = model(x_train_aug)  # (n_train+1, 1)
                residuals = torch.abs(y_train_aug - preds_aug).squeeze()  # (n_train+1,)
            rank = torch.sum(residuals[:n_train] <= residuals[n_train]).item() + 1
            for idx_siglvl in range(len(sig_lvls)):
                if rank <= conf_threshold[idx_siglvl]:
                    cp_set_all[idx_siglvl].append(y_train_aug[-1].item())
        cp_set_all = [np.array(cp_set) for cp_set in cp_set_all]
        # Take min/max (continuity ass.)
        cp_interval = np.array(
            [(cp_set.min(), cp_set.max()) for cp_set in cp_set_all]
        )  # shp (len(sig_lvls), 2)
        cp_intervals.append(cp_interval)

    cp_intervals = np.swapaxes(np.array(cp_intervals), 0, 1)  # shp (len(sig_lvls), n_test, 2)
    return cp_intervals


def run_acp(
    model: nn.Module,
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
    opt_prior_prec: float,
    opt_sigma_noise: float,
    sig_lvls: List[float],
    n_grid: int,
) -> NDArray:
    """Run the approximate full conformal prediction method of [Martinez et al., 2023].

    See the original paper "Approximating Full Conformal Prediction at Scale via Influence
    Functions" for details.

    Parameters
    ----------
    model: nn.Module
        Randomly initialized model.
    x_train: NDArray
        Input data used for training.
    y_train: NDArray
        Labels for the training data.
    x_test: NDArray
        Input data used for test.
    y_test: NDArray
        Labels for the test data.
    opt_prior_prec: float
        Optimal prior precision.
    opt_sigma_noise: float
        Optimal observation noise.se
    sig_lvls: List[float]
        List of significance levels to construct PI.
    n_grid: int
        Number of points in the grid used for postulated labels.

    Returns
    -------
    cp_intervals: NDArray
        The intervals construct by ACP.
    """
    sig_lvls = np.array(sig_lvls)
    y_grid = torch.linspace(y_train.min(), y_train.max(), n_grid, device=DEVICE)
    n_train = len(x_train)
    n_test = len(x_test)
    conf_threshold = np.ceil((n_train + 1) * (1 - sig_lvls)).astype(int)
    la = train_and_lap(model, opt_prior_prec, opt_sigma_noise, x_train, y_train, x_test, y_test)
    # Predictions on train data.
    jacobian, preds = la.backend.jacobians(x_train)
    # Reshape Jacobian needed for cross-leverage computation.
    n_batch, n_outs, n_params = jacobian.shape
    jacobian = jacobian.reshape(n_batch * n_outs, n_params)
    # "Leverage" and predictions on test points.
    jacobian_test, preds_test = la.backend.jacobians(x_test)
    h_test = la.functional_variance(jacobian_test)
    # "Cross-leverage" between test points and train data
    jacobian_test = jacobian_test.reshape(len(x_test) * n_outs, n_params)
    h_test_train = torch.einsum(
        "mp,pq,nq->mn", jacobian_test, la.posterior_covariance, jacobian
    )  # (M,N)
    h_test = h_test * (1 / opt_sigma_noise**2)
    h_test_train = h_test_train * (1 / opt_sigma_noise**2)
    # Scores on train points shp (M,n_grid,N).
    scores_train = torch.abs(
        (y_train - preds).squeeze()[None, None, :]
        - (y_grid.unsqueeze(0) - preds_test).unsqueeze(-1) * h_test_train.unsqueeze(1)
    )
    # Scores on test points shp (M,n_grid).
    scores_test = torch.abs(
        y_grid[None, :]
        - preds_test
        - (y_grid.unsqueeze(0) - preds_test) * h_test.squeeze()[:, None]
    )  # (M,n_grid)
    rank = torch.sum(scores_train <= scores_test[:, :, None], dim=-1) + 1  # shp (M,n_grid)
    rank = rank.squeeze().cpu().detach().numpy()
    y_grid = y_grid.squeeze().cpu().detach().numpy()
    grid_mask = np.expand_dims(rank, 0) <= np.expand_dims(
        conf_threshold, [1, 2]
    )  # of shape (len(sig_lvls), M, n_grid)
    cp_intervals = [[] for _ in range(len(sig_lvls))]
    for idx_siglvl in range(len(sig_lvls)):
        for idx_testpoint in range(n_test):
            mask = grid_mask[idx_siglvl, idx_testpoint]  # (n_grid,)
            cp_set = y_grid[mask]
            cp_intervals[idx_siglvl].append([cp_set.min(), cp_set.max()])
    cp_intervals = np.array(cp_intervals)  # shp (len(sig_lvls), n_test, 2)
    return cp_intervals


def run_experiment(
    method: str,
    sig_lvls: List[float],
    with_outliers: bool,
    crr_algo: str,
    split_calib_size: float,
    n_grid: int,
) -> Tuple[float, float, float]:
    """Train and evaluate a regression model on the synthetic dataset.

    The data can be generated with or without outliers. Apply the specified conformal prediction
    method and return the average interval size and coverage together with test statistics.

    Parameters
    ----------
    method: str
        CP method to use, either one of 'bayes', 'scp', 'acp', 'acpgn' or 'fullcp'.
    sig_lvls: List[float]
        List of significance levels to construct PI.
    with_outliers: bool
        Whether to include outliers in the synthetic dataset.
    crr_algo: str
        CRR implementation choice of "nouretdinov", "vovk", "vovk_mod", "burnaev".
    split_calib_size: float
        Fraction of the data to be reserved for calibration.
    n_grid: int
        Number of points in the grid used for postulated labels.

    Returns
    -------
    interval_width: float
        Average interval width on the test data.
    coverage: float
        Observed coverage on the test data.
    timetaken: float
        Time taken to complete the experiment.
    """
    # Generate data and setup dataloader.
    x_train, y_train, x_test, y_test = generate_synthetic_data(with_outliers)
    ds_train = data_utils.TensorDataset(x_train, y_train)
    if method == "scp":
        ds_train, ds_calib = data_utils.random_split(
            ds_train, [1 - split_calib_size, split_calib_size]
        )
    train_loader = data_utils.DataLoader(
        ds_train, batch_size=len(x_train), shuffle=True
    )  # Full batch training.
    # Simple 1-hidden layer Tanh MLP.
    model = nn.Sequential(nn.Linear(1, 100), nn.Tanh(), nn.Linear(100, 1)).to(DEVICE)
    print("Param count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Joint marglik-MAP training for purpose of obtaining optimal hyperparameters (not timed).
    print("joint marglik-MAP training")
    n_epochs = 5000
    la, model, _, _ = marglik_training(
        model=model,
        train_loader=train_loader,
        likelihood="regression",
        hessian_structure="full",
        backend=AsdlGGN,
        prior_structure="scalar",
        n_epochs=n_epochs,
        optimizer_cls=Adam,
        optimizer_kwargs={"lr": 1e-2},
        scheduler_cls=CosineAnnealingLR,
        scheduler_kwargs={"T_max": n_epochs, "eta_min": 1e-5},
        lr_hyp=0.1,
        n_epochs_burnin=100,
        n_hypersteps=50,
        marglik_frequency=50,
        prior_prec_init=1e-3,
        sigma_noise_init=1.0,
        temperature=1.0,
        fix_sigma_noise=False,
    )
    model.eval()
    with torch.no_grad():
        preds_test = model(x_test)
        test_mse = (preds_test.squeeze() - y_test.squeeze()).square().sum().item() / len(y_test)
        test_nll = -Normal(loc=preds_test, scale=1.0).log_prob(y_test).sum().item() / len(y_test)
    print(f"test_mse {test_mse}")
    print(f"test_nll {test_nll}")

    opt_prior_prec = la.prior_precision[0].item()
    opt_sigma_noise = la.sigma_noise.item()
    del la
    reset_parameters(model)
    print(
        f"sigma={opt_sigma_noise:.4f}",
        f"prior precision={opt_prior_prec:.4f}",
    )
    t0 = time.time()
    # cp_intervals shp (len(sig_lvls), n_test, 2)
    if method == "bayes":
        cp_intervals = run_bayes(
            model, x_train, y_train, x_test, y_test, opt_prior_prec, opt_sigma_noise, sig_lvls
        )
    elif method == "acpgn":
        cp_intervals = run_acpgn(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            opt_prior_prec,
            opt_sigma_noise,
            sig_lvls,
            crr_algo,
        )
    elif method == "scp":
        cp_intervals = run_scp(
            model,
            ds_train[:][0],
            ds_train[:][1],
            ds_calib[:][0],
            ds_calib[:][1],
            x_test,
            y_test,
            opt_prior_prec,
            opt_sigma_noise,
            sig_lvls,
        )
    elif method == "fullcp":
        cp_intervals = run_fullcp(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            opt_prior_prec,
            opt_sigma_noise,
            sig_lvls,
            n_grid,
        )
    elif method == "acp":
        cp_intervals = run_acp(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            opt_prior_prec,
            opt_sigma_noise,
            sig_lvls,
            n_grid,
        )
    else:
        raise ValueError("Invalid method")
    timetaken = time.time() - t0
    print(f"time: {timetaken}")
    interval_width = np.mean(
        cp_intervals[:, :, 1] - cp_intervals[:, :, 0], axis=-1
    )  # (len(sig_lvls),)
    y_test_np = y_test.squeeze().cpu().numpy()
    y_test_np = y_test_np.reshape((1, -1))
    coverage = np.mean(
        (y_test_np >= cp_intervals[:, :, 0]) & (y_test_np <= cp_intervals[:, :, 1]), axis=-1
    )
    return interval_width, coverage, timetaken


def main(config):
    """Main function that launches experiment."""
    save_dir = "results/synthetic/"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(config["seed"])  # Seed
    print("device", DEVICE)
    print(f"Seed {config['seed']}")
    print(f"Method {config['method']}")
    interval_width, coverage, timetaken = run_experiment(
        config["method"],
        config["sig_lvls"],
        config["outliers"],
        config["crr_algo"],
        config["split_calib_size"],
        config["n_grid"],
    )
    for kk, sig_lvl in enumerate(config["sig_lvls"]):
        print(f"\nCoverage: {100*(1-sig_lvl):.0f}%")
        print(f"cov={coverage[kk]:.3f}   width={interval_width[kk]:.3f}   time=({timetaken:.3f})")
    fname = (
        "synthetic"
        + ("_outlier" if config["outliers"] else "")
        + "_"
        + config["method"]
        + f"_{config['seed']}"
    )
    fn = os.path.join(save_dir, fname + ".npz")
    np.savez(
        fn,
        interval_width=interval_width,
        coverage=coverage,
        time=timetaken,
        method=config["method"],
        sig_lvls=config["sig_lvls"],
        seed=config["seed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default="acp", choices=["bayes", "acpgn", "scp", "fullcp", "acp"]
    )
    parser.add_argument("--split_calib_size", type=float, default=0.5)
    parser.add_argument("--sig_lvls", nargs="+", type=float, default=[0.1, 0.05, 0.01])
    parser.add_argument("--crr_algo", choices=CRR_ALGOS, default="burnaev")
    parser.add_argument("--outliers", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_grid", type=int, default=20)
    parser.set_defaults(outliers=False)
    cfg = parser.parse_args().__dict__
    main(cfg)
