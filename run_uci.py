# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/aleximmer/heteroscedastic-nn
# Copyright (c) 2025 Alex Immer, licensed under the MIT License
# License is provided for attribution purposes only, Not a Contribution

"""Script to run UCI regression experiments."""

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.utils.data as data_utils
import yaml
from laplace.utils import expand_prior_precision
from numpy.typing import NDArray
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset

from lib.crr import (
    CRR_ALGOS,
    FULL_CONF_METHODS,
    SPLIT_CONF_METHODS,
    eval_acp,
    eval_bayes,
    eval_cqr,
    eval_crr,
    eval_jackknife,
    eval_split_cp,
)
from lib.datasets_uci import (
    ALL_DATASETS,
    CQR_DATASETS,
    UCI_DATASETS,
    RegressionDatasets,
)
from lib.models import MLP
from lib.nn_training import train_quantile_regression_net, train_regression_net
from lib.trainer import train_laplace_model, train_laplace_model_calib_split
from lib.utils import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(
    seed: int,
    dataset: str,
    hyperparam_opt: str,
    cp_type: str,
    sig_lvls: List[float],
    crr_algo: str,
    split_calib_size: float,
    laplace_refine: bool = False,
    kfold_idx=-1,
) -> dict[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Train and evaluate a regression model.

    This is done in K-fold cross validation fashion if kfold_idx is provided.
    Apply (full or split) conformal prediction and return the average interval size and coverage
    together with test statistics.

    Parameters
    ----------
    seed: int
        Random seed for reproducibility.
    dataset: str
        Name of the dataset used in this experiment.
    hyperparam_opt: str
        Determines which type of hyperparameter optimization to use. Either 'marglik' or 'cv'.
    cp_type: str
        Either one of {'full', 'full_refine', 'split'}.
    sig_lvls: List[float]
        List of significance levels to construct PI (default=0.05).
    crr_algo: str
        CRR implementation choice of "nouretdinov", "vovk", "vovk_mod", "burnaev".
    split_calib_size: float
        Fraction of the data to be reserved for calibration.
    laplace_refine: bool
        Whether to run Laplace refinement.
    kfold_idx: int
        Which data fold to consider in this run. Set to -1 if not doing K-fold splits.

    Returns
    -------
    cp_intervals: NDArray
        The computed conformal prediction intervals.
    interval_widths_avg: NDArray
        The average size of the prediction interval on the test data.
    coverage: NDArray
        The observed coverage on the test data.
    test_mse: NDArray
        The mean squared error on the test data.
    test_loglik: NDArray
        The estimated log-likelihood of the test data for a fixed variance (observation noise).
    test_loglik_bayes: NDArray
        The estimated log-likelihood of the test data for variance given by Laplace approximation.
    """
    if cp_type == "full":
        # For full conformal prediction we do not need a calibration dataset
        split_calib_size = 0.0
    # Load config files
    if dataset in UCI_DATASETS:
        with open("configs/uci.yaml", "rb") as f:
            cfg = yaml.safe_load(f)
    elif dataset in CQR_DATASETS:
        with open("configs/cqr.yaml", "rb") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError("unrecognised dataset")

    # 90%/10% train/test split same as [Hernandez-Lobato & Adams, 2015]
    ds_kwargs = {
        "split_train_size": 0.9,
        "split_valid_size": 0.1,
        "split_calib_size": split_calib_size,
        "root": "data/",
        "seed": seed,
        "kfold_idx": kfold_idx,
        "device": DEVICE,
    }
    ds_train = RegressionDatasets(dataset, split="train", **ds_kwargs)
    ds_train_full = RegressionDatasets(
        dataset, split="train", **{**ds_kwargs, **{"split_valid_size": 0.0}}
    )
    ds_valid = RegressionDatasets(dataset, split="valid", **ds_kwargs)
    ds_test = RegressionDatasets(dataset, split="test", **ds_kwargs)
    ds_calib = None  # To be defined according to cp_type.
    if cp_type != "full":
        ds_calib = RegressionDatasets(dataset, split="calib", **ds_kwargs)

    train_loader = data_utils.DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    train_loader_full = data_utils.DataLoader(
        ds_train_full, batch_size=cfg["batch_size"], shuffle=True
    )
    data_loader_jacs = data_utils.DataLoader(
        ds_train_full, batch_size=cfg["batch_size"], shuffle=False
    )
    valid_loader = data_utils.DataLoader(ds_valid, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = data_utils.DataLoader(ds_test, batch_size=cfg["batch_size"], shuffle=False)
    if cp_type != "full":
        calib_loader = data_utils.DataLoader(ds_calib, batch_size=cfg["batch_size"], shuffle=False)
    if cp_type == "full_refine":
        # NB: Ensure "train set" for purposes of CRR is calibration set.
        data_loader_jacs = data_utils.DataLoader(
            ds_calib, batch_size=cfg["batch_size"], shuffle=False
        )

    # Single hidden layer MLP with GeLU activation function.
    input_size = ds_train_full.data.size(1)
    model = MLP(
        input_size, cfg["width"], cfg["depth"], output_size=1, activation=cfg["activation"]
    ).to(DEVICE)
    print("Param count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if cp_type != "full_refine":
        la, la_bayes = train_laplace_model(
            hyperparam_opt,
            model,
            train_loader_full,
            train_loader,
            valid_loader,
            cfg["n_epochs"],
            cfg["n_epochs_cv"],
            cfg["lr"],
            cfg["lr_cv"],
            cfg["lr_min"],
            cfg["lr_hyp"],
            cfg["lr_hyp_min"],
            cfg["n_epochs_burnin"],
            cfg["n_hypersteps"],
            cfg["marglik_frequency"],
            cfg["prior_prec_init"],
            cfg["prior_prec_log_lower"],
            cfg["prior_prec_log_upper"],
        )
    else:
        la, la_bayes = train_laplace_model_calib_split(
            hyperparam_opt,
            model,
            train_loader_full,
            train_loader,
            valid_loader,
            calib_loader,
            cfg["n_epochs"],
            cfg["n_epochs_cv"],
            cfg["lr"],
            cfg["lr_cv"],
            cfg["lr_min"],
            cfg["lr_hyp"],
            cfg["lr_hyp_min"],
            cfg["n_epochs_burnin"],
            cfg["n_hypersteps"],
            cfg["marglik_frequency"],
            cfg["prior_prec_init"],
            cfg["prior_prec_log_lower"],
            cfg["prior_prec_log_upper"],
        )

    # NB: code immediately below only needed for refinement by 1-step solve (when merge this with
    # kfac script).
    js = []
    preds = []
    for batch_x, batch_y in data_loader_jacs:
        js_batch, preds_batch = la.backend.jacobians(batch_x)
        preds_batch = preds_batch.squeeze()
        _, n_outs, n_params = js_batch.shape
        js_batch = js_batch.reshape(len(batch_x) * n_outs, n_params)
        js.append(js_batch)
        preds.append(preds_batch)
    js = torch.cat(js)  # (N,P)
    preds = torch.cat(preds)  # (N,)
    targets = ds_calib.targets if cp_type == "full_refine" else ds_train_full.targets

    # Split-CP baselines code.
    if cp_type == "split":
        # Normalized split-CP baseline.
        targets_crf = torch.log(
            torch.abs(targets - preds[:, None])
        ).detach()  # conformalized residual fitting targets
        inputs_crf = ds_train_full.data
        ds_crf = TensorDataset(inputs_crf, targets_crf)
        train_loader_full_crf = data_utils.DataLoader(
            ds_crf, batch_size=cfg["batch_size"], shuffle=True
        )
        model_crf = MLP(
            input_size,
            cfg["width"],
            cfg["depth"],
            output_size=1,
            activation=cfg["activation"],
        ).to(DEVICE)
        model_crf, _, _, _ = train_regression_net(
            model=model_crf,
            train_loader=train_loader_full_crf,
            prior_prec=expand_prior_precision(la.prior_precision, la.model) * la.sigma_noise**2,
            n_epochs=cfg["n_epochs"],
            scheduler_kwargs={"T_max": cfg["n_epochs"], "eta_min": cfg["lr_min"]},
            optimizer_kwargs={"lr": cfg["lr_cv"]},
            scheduler_cls=CosineAnnealingLR,
        )

        preds_calib_crf = []
        for batch_x, batch_y in calib_loader:
            preds_batch = model_crf(batch_x)
            preds_batch = preds_batch.squeeze()
            preds_calib_crf.append(preds_batch)
        preds_calib_crf = torch.cat(preds_calib_crf)

        preds_test_crf = []
        for batch_x, batch_y in test_loader:
            preds_batch = model_crf(batch_x)
            preds_batch = preds_batch.squeeze()
            preds_test_crf.append(preds_batch)
        preds_test_crf = torch.cat(preds_test_crf)

        # Conformal Quantile Regression (CQR) baseline.
        model_cqr = MLP(
            input_size,
            cfg["width"],
            cfg["depth"],
            output_size=2 * len(sig_lvls),
            activation=cfg["activation"],
        ).to(DEVICE)

        quantiles = list(
            sum([(sig_lvl / 2, 1 - sig_lvl / 2) for sig_lvl in sig_lvls], ())
        )  # quantiles

        model_cqr, _ = train_quantile_regression_net(
            model=model_cqr,
            train_loader=train_loader_full,
            prior_prec=expand_prior_precision(la.prior_precision, model_cqr) * la.sigma_noise**2,
            quantiles=quantiles,
            n_epochs=cfg["n_epochs"],
            scheduler_kwargs={"T_max": cfg["n_epochs"], "eta_min": cfg["lr_min"]},
            optimizer_kwargs={"lr": cfg["lr_cv"]},
            scheduler_cls=CosineAnnealingLR,
        )

        preds_calib_cqr = []
        for batch_x, batch_y in calib_loader:
            preds_batch = model_cqr(batch_x)
            preds_calib_cqr.append(preds_batch)
        preds_calib_cqr = torch.cat(preds_calib_cqr)  # shp (N_cal, len(sig_lvls)*2)
        preds_calib_cqr = preds_calib_cqr.reshape(
            (-1, len(sig_lvls), 2)
        )  # shp (N_cal, len(sig_lvls), 2)

        preds_test_cqr = []
        for batch_x, batch_y in test_loader:
            preds_batch = model_cqr(batch_x)
            preds_test_cqr.append(preds_batch)
        preds_test_cqr = torch.cat(preds_test_cqr)  # shp (N_test, len(sig_lvls)*2)
        preds_test_cqr = preds_test_cqr.reshape(
            (-1, len(sig_lvls), 2)
        )  # shp (N_test, len(sig_lvls), 2)

    # do refinement for both `la` and `la_bayes`
    theta_star = parameters_to_vector(la.model.parameters()).detach()
    f_offset = preds - (js @ theta_star)
    theta_bayes_refine = (
        (1 / la_bayes.sigma_noise.square())
        * la_bayes.posterior_covariance
        @ js.T
        @ (targets.squeeze() - f_offset)
    )
    theta_refine = (
        (1 / la.sigma_noise.square())
        * la.posterior_covariance
        @ js.T
        @ (targets.squeeze() - f_offset)
    )
    # Evaluate functional variances on test set for Laplace.
    # Report MAP and posterior predictive performance.
    n_test = len(test_loader.dataset)
    scale = ds_train_full.s  # transform back to original targets when reporting performance
    test_mse = 0
    test_loglik = 0
    test_loglik_bayes = 0
    test_loglik_bayes_refine = 0
    fvar_test = []
    for batch_x, batch_y in test_loader:
        js_batch, preds_batch = la_bayes.backend.jacobians(batch_x)
        preds_batch = preds_batch.squeeze()
        fvar = la_bayes.functional_variance(js_batch).squeeze()
        fvar_test.append(fvar)
        _, n_outs, n_params = js_batch.shape
        js_batch = js_batch.reshape(len(batch_x) * n_outs, n_params)
        preds_refine_batch = preds_batch + (js_batch @ (theta_bayes_refine - theta_star))

        test_mse += (preds_batch - batch_y.squeeze()).square().sum().item() / n_test
        pred_dist = Normal(loc=preds_batch * scale, scale=la_bayes.sigma_noise * scale)
        test_loglik += pred_dist.log_prob(batch_y.squeeze() * scale).sum().item() / n_test
        y_std = torch.sqrt(fvar + la_bayes.sigma_noise.item() ** 2)
        pred_dist = Normal(loc=preds_batch * scale, scale=y_std * scale)
        test_loglik_bayes += pred_dist.log_prob(batch_y.squeeze() * scale).sum().item() / n_test
        pred_dist = Normal(loc=preds_refine_batch * scale, scale=y_std * scale)
        test_loglik_bayes_refine += (
            pred_dist.log_prob(batch_y.squeeze() * scale).sum().item() / n_test
        )
    fvar_test = torch.cat(fvar_test)
    print(
        f"Test performance: MSE={test_mse:.3f}, LogLik={test_loglik:.3f}, "
        f"LogLikBayes={test_loglik_bayes:.3f}, LogLikBayesRefine={test_loglik_bayes_refine:.3f}"
    )
    # Evaluate quantities needed for CRR methods.
    preds_test = []
    preds_refine_test = []
    preds_refine_bayes_test = []
    js_test = []
    h_test = []
    for batch_x, batch_y in test_loader:
        # Evaluate "leverage" and predictions on test points.
        js_batch, preds_batch = la.backend.jacobians(batch_x)
        preds_batch = preds_batch.squeeze()
        fvar = la.functional_variance(js_batch).squeeze()
        _, n_outs, n_params = js_batch.shape
        # and cache Jacs to compute cross-leverages later.
        js_batch = js_batch.reshape(len(batch_x) * n_outs, n_params)
        preds_refine_batch = preds_batch + (js_batch @ (theta_refine - theta_star))
        preds_refine_bayes_batch = preds_batch + (js_batch @ (theta_bayes_refine - theta_star))
        preds_test.append(preds_batch)
        preds_refine_test.append(preds_refine_batch)
        preds_refine_bayes_test.append(preds_refine_bayes_batch)
        js_test.append(js_batch)
        h_test.append(fvar)
    preds_test = torch.cat(preds_test)
    preds_refine_test = torch.cat(preds_refine_test)
    preds_refine_bayes_test = torch.cat(preds_refine_bayes_test)
    js_test = torch.cat(js_test)
    h_test = torch.cat(h_test)
    if cp_type in ["full", "full_refine"]:
        preds = []
        preds_refine = []
        h_train = []
        h_test_train = []
        for batch_x, batch_y in data_loader_jacs:
            # Leverage and predictions on train data.
            js_batch, preds_batch = la.backend.jacobians(batch_x)
            preds_batch = preds_batch.squeeze()
            fvar = la.functional_variance(js_batch).squeeze()
            _, n_outs, n_params = js_batch.shape
            js_batch = js_batch.reshape(len(batch_x) * n_outs, n_params)
            fvar_cross = torch.einsum(
                "mp,pq,nq->mn", js_test, la.posterior_covariance, js_batch
            )  # (M,batch_size)
            preds_refine_batch = preds_batch + (js_batch @ (theta_refine - theta_star))
            preds.append(preds_batch)
            preds_refine.append(preds_refine_batch)
            h_train.append(fvar)
            h_test_train.append(fvar_cross)
        preds = torch.cat(preds)
        preds_refine = torch.cat(preds_refine)
        h_train = torch.cat(h_train)
        h_test_train = torch.cat(h_test_train, dim=-1)
        if laplace_refine:
            preds_train = preds_refine.squeeze().cpu().detach().numpy()
            preds_eval = preds_refine_test.squeeze().cpu().detach().numpy()
        else:
            preds_train = preds.squeeze().cpu().detach().numpy()
            preds_eval = preds_test.squeeze().cpu().detach().numpy()
        intvl_kwargs = {
            "ys": targets.squeeze().cpu().detach().numpy(),
            "preds": preds_train,
            "preds_eval": preds_eval,
            "h_mn": h_test_train.squeeze().cpu().detach().numpy(),
            "h_n": h_train.squeeze().cpu().detach().numpy(),
            "sigma_sq": la.sigma_noise.item() ** 2,
            "sig_lvls": sig_lvls,
        }
        cp_intervals = []
        for method in FULL_CONF_METHODS:
            method_parse = method.split("_")
            if method == "bayes":
                # Only use refined predictions for bayes when fitted using marglik.
                # Observed poor performance otherwise, hunch that caused by MLE fit of sigma_noise.
                if laplace_refine and (hyperparam_opt == "marglik"):
                    fmu = preds_refine_bayes_test.squeeze().cpu().detach().numpy()
                else:
                    fmu = preds_test.squeeze().cpu().detach().numpy()

                cp_intervals += [
                    eval_bayes(
                        fmu=fmu,
                        fvar=fvar_test.squeeze().cpu().detach().numpy(),
                        sigma_sq=la_bayes.sigma_noise.item() ** 2,
                        sig_lvls=sig_lvls,
                    )
                ]
            elif method_parse[0] == "crr":
                cp_intervals += [
                    eval_crr(
                        h_m=h_test.squeeze().cpu().detach().numpy(),
                        algo=crr_algo,
                        nonconformity_score=method_parse[1],
                        **intvl_kwargs,
                    )
                ]
            elif method_parse[0] == "jackknife":
                cp_intervals += [eval_jackknife(method=method_parse[1], **intvl_kwargs)]
            elif method == "acp":
                cp_intervals += [
                    eval_acp(h_m=h_test.squeeze().cpu().detach().numpy(), **intvl_kwargs)
                ]
            else:
                raise ValueError("Invalid method")
    else:
        preds_calib = []
        h_calib = []
        for batch_x, batch_y in calib_loader:
            # evaluate "leverage" and predictions on calibration points
            js_batch, preds_batch = la.backend.jacobians(batch_x)
            preds_batch = preds_batch.squeeze()
            fvar = la.functional_variance(js_batch).squeeze()

            preds_calib.append(preds_batch)
            h_calib.append(fvar)
        preds_calib = torch.cat(preds_calib)
        h_calib = torch.cat(h_calib)

        intvl_kwargs = {
            "ys_calib": ds_calib.targets.squeeze().cpu().detach().numpy(),
            "preds_calib": preds_calib.squeeze().cpu().detach().numpy(),
            "preds_eval": preds_test.squeeze().cpu().detach().numpy(),
            "sigma_sq": la.sigma_noise.item() ** 2,
            "sig_lvls": sig_lvls,
        }
        cp_intervals = []
        for method in SPLIT_CONF_METHODS:
            method_parse = method.split("_", 1)
            if method_parse[1] == "standard":
                cp_intervals += [eval_split_cp(**intvl_kwargs)]
            elif method_parse[1] in ["norm_var", "norm_std"]:
                cp_intervals += [
                    eval_split_cp(
                        h_eval=h_test.squeeze().cpu().detach().numpy(),
                        h_calib=h_calib.squeeze().cpu().detach().numpy(),
                        nonconformity_score=method_parse[1],
                        **intvl_kwargs,
                    )
                ]
            elif method_parse[1] == "crf":
                cp_intervals += [
                    eval_split_cp(
                        norm_eval=torch.exp(preds_test_crf).squeeze().cpu().detach().numpy(),
                        norm_calib=torch.exp(preds_calib_crf).squeeze().cpu().detach().numpy(),
                        nonconformity_score=method_parse[1],
                        **intvl_kwargs,
                    )
                ]
            elif method_parse[1] == "cqr":
                cp_intervals += [
                    eval_cqr(
                        ys_calib=ds_calib.targets.squeeze().cpu().detach().numpy(),
                        preds_calib=preds_calib_cqr.squeeze().cpu().detach().numpy(),
                        preds_eval=preds_test_cqr.squeeze().cpu().detach().numpy(),
                        sig_lvls=sig_lvls,
                    )
                ]
            else:
                raise ValueError("Invalid method")

    cp_intervals = np.array(cp_intervals)  # (n_methods, len(sig_lvls), n_test, 2)
    # Mean PI width adjusted by targets standardization from data-preprocessing step.
    interval_widths_avg = np.mean(
        scale * (cp_intervals[:, :, :, 1] - cp_intervals[:, :, :, 0]), axis=-1
    )  # (n_methods, len(sig_lvls))

    # Coverage.
    y_test_np = ds_test.targets.squeeze().cpu().numpy()
    y_test_np = y_test_np.reshape((1, 1, -1))
    coverage = np.mean(
        (y_test_np >= cp_intervals[:, :, :, 0]) & (y_test_np <= cp_intervals[:, :, :, 1]), axis=-1
    )
    return {
        "cp_intervals": cp_intervals * scale,
        "interval_widths_avg": interval_widths_avg,
        "coverage": coverage,
        "test_mse": np.array(test_mse),
        "test_loglik": np.array(test_loglik),
        "test_loglik_bayes": np.array(test_loglik_bayes),
    }


def main(config):
    """Main function that launches experiment."""
    # Create folder to save results.
    save_dir = f"results/uci/{config['dataset']}/"
    os.makedirs(save_dir, exist_ok=True)
    # Set random seed.
    set_seed(config["seed"])
    print(f"Running on {DEVICE}.")
    # Whether to run Laplace refinement.
    laplace_refine = config["cp_type"] == "full_refine"
    if config["kfold"]:
        # Run experiment with 10 different folds.
        results_all = []
        for fold in range(10):
            print(f"Seed {config['seed']}  Fold {fold+1}")
            results_all.append(
                run_experiment(
                    config["seed"],
                    config["dataset"],
                    config["hyperparam_opt"],
                    config["cp_type"],
                    config["sig_lvls"],
                    config["crr_algo"],
                    config["split_calib_size"],
                    laplace_refine,
                    fold,
                )
            )
        results = {
            k: np.array([dd[k] for dd in results_all])
            for k in results_all[0]
            if k != "cp_intervals"
        }
        # For cp_intervals just take last fold.
        results["cp_intervals"] = results_all[-1][
            "cp_intervals"
        ]  # (n_methods, len(sig_lvls), n_test, 2)

        results["interval_widths_avg"] = np.moveaxis(
            results["interval_widths_avg"], 0, -1
        )  # (n_methods, len(sig_lvls), n_repeats)
        results["coverage"] = np.moveaxis(
            results["coverage"], 0, -1
        )  # (n_methods, len(sig_lvls), n_repeats)
    else:
        print(f"Seed {config['seed']}")
        results = run_experiment(
            config["seed"],
            config["dataset"],
            config["hyperparam_opt"],
            config["cp_type"],
            config["sig_lvls"],
            config["crr_algo"],
            config["split_calib_size"],
            laplace_refine,
        )
        # Ensure consistent dims to above.
        results["interval_widths_avg"] = np.expand_dims(results["interval_widths_avg"], -1)
        results["coverage"] = np.expand_dims(results["coverage"], -1)
    n_repeats = results["test_mse"].size
    print(
        f"\nMSE={results['test_mse'].mean():.3f} "
        f"({results['test_mse'].std()/np.sqrt(n_repeats):.3f}), "
        f"LogLik={results['test_loglik'].mean():.3f} "
        f"({results['test_loglik'].std()/np.sqrt(n_repeats):.3f}), "
        f"LogLikBayes={results['test_loglik_bayes'].mean():.3f} "
        f"({results['test_loglik_bayes'].std()/np.sqrt(n_repeats):.3f})"
    )
    methods = (
        FULL_CONF_METHODS if config["cp_type"] in ["full", "full_refine"] else SPLIT_CONF_METHODS
    )
    print()
    std_norm = np.sqrt(n_repeats)
    for kk, sig_lvl in enumerate(config["sig_lvls"]):
        print(f"Coverage: {100*(1-sig_lvl):.0f}%")
        for jj, method in enumerate(methods):
            print(
                f"{method:18s}: cov={results['coverage'].mean(axis=-1)[jj,kk].item():.3f} "
                f"({results['coverage'].std(axis=-1)[jj,kk].item()/std_norm:.3f})  "
                f"width={results['interval_widths_avg'].mean(axis=-1)[jj,kk].item():.3f} "
                f"({results['interval_widths_avg'].std(axis=-1)[jj,kk].item()/std_norm:.3f})"
            )
        print()
    fname = f"{config['dataset']}_{config['hyperparam_opt']}"
    if config["cp_type"] == "split":
        fname += f"_split{int(config['split_calib_size']*100)}"
    elif config["cp_type"] == "full_refine":
        fname += f"_full{int(config['split_calib_size']*100)}"
    fname += f"_{config['seed']}"
    fn = os.path.join(save_dir, fname + ".npz")
    np.savez(
        fn,
        methods=methods,
        sig_lvls=config["sig_lvls"],
        split_calib_size=config["split_calib_size"],
        **results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="boston", choices=ALL_DATASETS)
    parser.add_argument("--cp_type", default="full", choices=["full", "full_refine", "split"])
    parser.add_argument("--split_calib_size", type=float, default=0.5)
    parser.add_argument("--sig_lvls", nargs="+", type=float, default=[0.1, 0.05, 0.01])
    parser.add_argument("--crr_algo", choices=CRR_ALGOS, default="burnaev")
    parser.add_argument("--hyperparam_opt", default="marglik", choices=["cv", "marglik"])
    parser.add_argument("--kfold", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(kfold=False)
    config = parser.parse_args().__dict__
    main(config)
