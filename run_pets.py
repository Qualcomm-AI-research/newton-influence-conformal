# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Script to run Oxford Pets experiment."""

import argparse
import copy
import os
from typing import List, Optional

import numpy as np
import torch
import yaml
from laplace.curvature.asdl import AsdlGGN
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

from lib.bounding_box import (
    BBoxPredictor,
    PetDataset,
    PetDatasetResiduals,
    do_evalbatch_bbox_clf,
    get_backbone,
    train_bbox_net,
)
from lib.crr import (
    FULL_CONF_METHODS,
    SPLIT_CONF_METHODS,
    eval_bayes_multioutput,
    eval_conf_hyperrectangle_metrics,
    eval_cqr_multioutput,
    eval_crr_multioutput,
    eval_split_cp_multioutput,
)
from lib.marglik_training import marglik_training_posthoc
from lib.utils import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(
    seed: int,
    cp_type: str,
    sig_lvls: List[float],
    split_calib_size: float,
    backbone: str,
    laplace_refine: bool = False,
    model_ckpt_dir: Optional[str] = None,
) -> dict:
    """Train and evaluate a bounding box prediction model.

    Apply (full or split) conformal prediction methods and return the average interval size and
    coverage together with test statistics.

    Parameters
    ----------
    seed: int
        Random seed.
    cp_type: str
        Type of conformal prediction algorithm to run.
        Either one of {`full`, `full_refine`, `split`}.
    sig_lvls: List[float]
        List of significance levels to construct PI.
    split_calib_size: float
        Fraction of the data to be reserved for calibration.
    backbone: str
        Indicates the type of backbone to be used.
    laplace_refine: bool
        Whether to run Laplace refinement.
    model_ckpt_dir: str
        Path to the directory where model checkpoints will be saved.

    Returns
    -------
    volumes: NDArray
        The average volume of prediction per conformal method; 5 methods if cp_type = 'split' and
        4 methods if cp_type='full' or 'full_refine'.
    coverage: NDArray
        The average coverage of prediction per conformal method; 5 methods if cp_type = 'split' and
        4 methods if cp_type='full' or 'full_refine'.
    test_loc_error:
        Localization error (IOU) on the test data.
    test_acc: float
        Classification accuracy on the test data.
    """
    if cp_type == "full":
        split_calib_size = 0.0
    save_model = False
    if model_ckpt_dir is not None:
        save_model = True
    # Get config variables from config file.
    with open(f"configs/pet_{backbone}.yaml", "rb") as f:
        cfg = yaml.safe_load(f)
    # Load bounding box annotations.
    csv_file = "data/pet_bb_annotations.csv"
    ds_kwargs = {
        "csv_file": csv_file,
        "split_train_size": 0.8,
        "split_calib_size": split_calib_size,
        "device": DEVICE,
        "seed": seed,
    }
    # Create datasets and data loaders.
    train_dataset = PetDataset(split="train", data_aug=cfg["data_aug"], **ds_kwargs)
    test_dataset = PetDataset(split="test", **ds_kwargs)
    batch_size = cfg["batch_size"] if cp_type == "full" else cfg["batch_size_split"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Get backbone model.
    base_model, n_features = get_backbone(backbone)
    model = BBoxPredictor(base_model, n_features).to(DEVICE)
    # Train and (maybe) save the model.
    model = train_bbox_net(model, train_loader, n_epochs=cfg["n_epochs"], lr=cfg["lr"])
    if save_model:
        fname_mdl = f"{cp_type.replace('_','')}"
        if cp_type != "full":
            fname_mdl += f"{int(split_calib_size*100)}"
        fname_mdl += f"_{seed}.pt"
        torch.save(model.state_dict(), os.path.join(model_ckpt_dir, "model_" + fname_mdl))
    model.eval()
    # Evaluate the model.
    n_test = len(test_loader.dataset)
    with torch.no_grad():
        loc_error = 0
        acc = 0
        for x, y_bbox, y_lbl in test_loader:
            f_bbox, f_logit = model(x)
            loc_error_batch, acc_batch = do_evalbatch_bbox_clf(y_bbox, y_lbl, f_bbox, f_logit)
            loc_error += loc_error_batch * len(x)
            acc += acc_batch * len(x)
        loc_error /= n_test
        acc /= n_test
    print(f"test-loc-error={loc_error:.3f}  test-acc={acc:.3f}")

    # Forward hook to force model to return only bbox prediction (necessary for Laplace package).
    def filter_output(module, input, output):
        return output[0]

    model.register_forward_hook(filter_output)
    # Override datasets and loaders (bbox targets only).
    train_dataset = PetDataset(split="train", with_labels=False, **ds_kwargs)
    test_dataset = PetDataset(split="test", with_labels=False, **ds_kwargs)
    if cp_type != "full":
        calib_dataset = PetDataset(split="calib", with_labels=False, **ds_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size_jacs"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size_jacs"], shuffle=False)
    data_loader_jacs = train_loader
    if cp_type != "full":
        calib_loader = DataLoader(calib_dataset, batch_size=cfg["batch_size_jacs"], shuffle=False)
    if cp_type == "full_refine":
        data_loader_jacs = calib_loader
    # Train the marginal likelihood estimator.
    la, model, _ = marglik_training_posthoc(
        model=model,
        train_loader=data_loader_jacs,
        likelihood="regression",
        hessian_structure="full",
        backend=AsdlGGN,
        prior_structure="scalar",
        n_steps=cfg["n_epochs_hyp"],
        lr_hyp=cfg["lr_hyp"],
        lr_hyp_min=cfg["lr_hyp_min"],
        prior_prec_init=cfg["prior_prec_init"],
        sigma_noise_init=1.0,
        temperature=1.0,
        fix_sigma_noise=False,
        subset_of_weights="last_layer",
        laplace_kwargs={"last_layer_name": "bbox"},
    )
    # Apply Laplace refinement.
    Js = []
    targets = []
    preds = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader_jacs:
            Js_batch, preds_batch = la.backend.last_layer_jacobians(batch_x)
            _, n_outs, n_params = Js_batch.shape
            Js_batch = Js_batch.reshape(len(batch_x) * n_outs, n_params)
            Js.append(Js_batch)
            targets.append(batch_y)
            preds.append(preds_batch)
    Js = torch.cat(Js)  # (NK,P)
    targets = torch.cat(targets)  # (N,K)
    preds = torch.cat(preds)  # (N,K)
    if cp_type == "split":
        # Normalized split-CP baseline.
        targets_crf = torch.log(
            torch.abs(targets - preds)
        ).detach()  # Conformalized residual fitting targets
        ds_crf = PetDatasetResiduals(csv_file, train_dataset.indices, targets_crf, DEVICE)
        train_loader_crf = DataLoader(ds_crf, batch_size=cfg["batch_size_split"], shuffle=True)

        # Warmstart CRF network to original network but reset bbox last-layer parameters.
        base_model, n_features = get_backbone(backbone)
        model_crf = BBoxPredictor(
            base_model, n_features, with_clf=False, freeze_base_model=True
        ).to(DEVICE)
        model_state_dict = copy.deepcopy(model.state_dict())
        model_state_dict.pop("clf.weight", None)  # Remove parameters related to classifier.
        model_state_dict.pop("clf.bias", None)
        model_crf.load_state_dict(model_state_dict)
        model_crf.bbox.reset_parameters()
        model_crf = train_bbox_net(
            model_crf,
            train_loader_crf,
            n_epochs=cfg["n_epochs"] // 2,
            lr=cfg["lr"],
            with_warmup=False,
            with_clf=False,
        )
        if save_model:
            torch.save(
                model_crf.state_dict(), os.path.join(model_ckpt_dir, "model_crf_" + fname_mdl)
            )
        model_crf.eval()
        preds_calib_crf = []
        preds_test_crf = []
        with torch.no_grad():
            for batch_x, _ in calib_loader:
                preds_batch = model_crf(batch_x)
                preds_calib_crf.append(preds_batch)
            preds_calib_crf = torch.cat(preds_calib_crf)

            for batch_x, _ in test_loader:
                preds_batch = model_crf(batch_x)
                preds_test_crf.append(preds_batch)
            preds_test_crf = torch.cat(preds_test_crf)
        del model_crf
        torch.cuda.empty_cache()
        # CQR baseline
        base_model, n_features = get_backbone(backbone)
        model_cqr = BBoxPredictor(
            base_model,
            n_features,
            with_clf=False,
            freeze_base_model=True,
            n_quantiles=2 * len(sig_lvls),
        ).to(DEVICE)
        # Warmstart backbone of CQR network to SCP trained network.
        model_state_dict.pop("bbox.bias", None)
        model_state_dict.pop("bbox.weight", None)
        model_cqr.load_state_dict(model_state_dict, strict=False)
        quantiles = list(sum([(sig_lvl / 2, 1 - sig_lvl / 2) for sig_lvl in sig_lvls], ()))
        train_loader_cqr = DataLoader(
            train_dataset, batch_size=cfg["batch_size_split"], shuffle=True
        )
        test_loader_cqr = DataLoader(
            test_dataset, batch_size=cfg["batch_size_split"], shuffle=False
        )
        model_cqr = train_bbox_net(
            model_cqr,
            train_loader_cqr,
            n_epochs=cfg["n_epochs"] // 2,
            lr=cfg["lr"],
            with_warmup=False,
            with_clf=False,
            quantiles=quantiles,
        )
        if save_model:
            torch.save(
                model_cqr.state_dict(), os.path.join(model_ckpt_dir, "model_cqr_" + fname_mdl)
            )
        model_cqr.eval()
        preds_calib_cqr = []
        for batch_x, batch_y in calib_loader:
            preds_batch = model_cqr(batch_x)
            preds_calib_cqr.append(preds_batch)
        preds_calib_cqr = torch.cat(preds_calib_cqr)  # shp (N_cal, K, len(sig_lvls)*2)
        preds_calib_cqr = preds_calib_cqr.reshape(
            (-1, 4, len(sig_lvls), 2)
        )  # with shape (N_cal, K, len(sig_lvls), 2)
        preds_test_cqr = []
        for batch_x, batch_y in test_loader:
            preds_batch = model_cqr(batch_x)
            preds_test_cqr.append(preds_batch)
        preds_test_cqr = torch.cat(preds_test_cqr)  # shp (N_test, K, len(sig_lvls)*2)
        preds_test_cqr = preds_test_cqr.reshape(
            (-1, 4, len(sig_lvls), 2)
        )  # shp (N_test, K, len(sig_lvls), 2)
        del model_cqr
        torch.cuda.empty_cache()
    targets = targets.reshape(-1)  # (NK,)
    theta_star = parameters_to_vector(la.model.last_layer.parameters()).detach()
    theta_refine = (1 / la.sigma_noise.square()) * la.posterior_covariance @ Js.T @ targets
    # Evaluate quantities needed for CRR methods.
    targets_test = []
    preds_test = []
    preds_refine_test = []
    Js_test = []
    h_test = []
    for batch_x, batch_y in test_loader:
        # evaluate "leverage" and predictions on test points
        Js_batch, preds_batch = la.backend.last_layer_jacobians(batch_x)
        fvar = la.functional_variance(Js_batch)
        preds_refine_batch = preds_batch + torch.einsum(
            "mcp,p->mc", Js_batch, theta_refine - theta_star
        )

        targets_test.append(batch_y)
        preds_test.append(preds_batch)
        preds_refine_test.append(preds_refine_batch)
        Js_test.append(Js_batch)
        h_test.append(fvar)
    targets_test = torch.cat(targets_test)
    preds_test = torch.cat(preds_test)
    preds_refine_test = torch.cat(preds_refine_test)
    Js_test = torch.cat(Js_test)
    h_test = torch.cat(h_test)
    if cp_type in ["full", "full_refine"]:
        targets = []
        preds = []
        preds_refine = []
        h_train = []
        h_test_train = []
        for batch_x, batch_y in data_loader_jacs:
            # leverage and predictions on train data
            Js_batch, preds_batch = la.backend.last_layer_jacobians(batch_x)
            fvar = la.functional_variance(Js_batch)
            preds_refine_batch = preds_batch + torch.einsum(
                "mcp,p->mc", Js_batch, theta_refine - theta_star
            )

            fvar_cross = torch.einsum(
                "mcp,pq,nkq->mnck", Js_test, la.posterior_covariance, Js_batch
            )  # (M,N,K,K)

            targets.append(batch_y)
            preds.append(preds_batch)
            preds_refine.append(preds_refine_batch)
            h_train.append(fvar)
            h_test_train.append(fvar_cross)
        targets = torch.cat(targets)
        preds = torch.cat(preds)
        preds_refine = torch.cat(preds_refine)
        h_train = torch.cat(h_train)
        h_test_train = torch.cat(h_test_train, dim=1)
        if laplace_refine:
            preds_train = preds_refine.cpu().detach().numpy()
            preds_eval = preds_refine_test.cpu().detach().numpy()
        else:
            preds_train = preds.cpu().detach().numpy()
            preds_eval = preds_test.cpu().detach().numpy()
        intvl_kwargs = {
            "ys": targets.cpu().detach().numpy(),
            "preds": preds_train,
            "preds_eval": preds_eval,
            "h_m": h_test.cpu().detach().numpy(),
            "h_mn": h_test_train.cpu().detach().numpy(),
            "h_n": h_train.cpu().detach().numpy(),
            "sigma_sq": la.sigma_noise.item() ** 2,
            "sig_lvls": sig_lvls,
        }
        volumes = []
        coverage = []
        for method in FULL_CONF_METHODS:
            method_parse = method.split("_")
            if method == "bayes":
                vol, cov = eval_bayes_multioutput(
                    fmu=preds_eval,
                    fcov=h_test.cpu().detach().numpy(),
                    sigma_sq=la.sigma_noise.item() ** 2,
                    sig_lvls=sig_lvls,
                    ys=targets_test.cpu().detach().numpy(),
                )
                volumes += [vol]  # (len(sig_lvls),)
                coverage += [cov]
            elif method_parse[0] == "crr":
                cp_regions = eval_crr_multioutput(
                    nonconformity_score=method_parse[1], output_independence=True, **intvl_kwargs
                )
                vol, cov = eval_conf_hyperrectangle_metrics(cp_regions, targets_test.cpu().numpy())
                volumes += [vol]
                coverage += [cov]
            else:
                raise ValueError("Invalid method")
    else:
        targets_calib = []
        preds_calib = []
        h_calib = []
        for batch_x, batch_y in calib_loader:
            # evaluate "leverage" and predictions on calibration points
            Js_batch, preds_batch = la.backend.last_layer_jacobians(batch_x)
            fvar = la.functional_variance(Js_batch)

            targets_calib.append(batch_y)
            preds_calib.append(preds_batch)
            h_calib.append(fvar)
        targets_calib = torch.cat(targets_calib)
        preds_calib = torch.cat(preds_calib)
        h_calib = torch.cat(h_calib)

        intvl_kwargs = {
            "ys_calib": targets_calib.cpu().detach().numpy(),
            "preds_calib": preds_calib.cpu().detach().numpy(),
            "preds_eval": preds_test.cpu().detach().numpy(),
            "sigma_sq": la.sigma_noise.item() ** 2,
            "sig_lvls": sig_lvls,
        }
        cp_regions = []
        for method in SPLIT_CONF_METHODS:
            method_parse = method.split("_", 1)
            if method_parse[1] == "standard":
                cp_regions += [eval_split_cp_multioutput(**intvl_kwargs)]
            elif method_parse[1] in ["norm_var", "norm_std"]:
                cp_regions += [
                    eval_split_cp_multioutput(
                        h_eval=h_test.cpu().detach().numpy(),
                        h_calib=h_calib.cpu().detach().numpy(),
                        nonconformity_score=method_parse[1],
                        **intvl_kwargs,
                    )
                ]
            elif method_parse[1] == "crf":
                cp_regions += [
                    eval_split_cp_multioutput(
                        norm_eval=torch.exp(preds_test_crf).cpu().detach().numpy(),
                        norm_calib=torch.exp(preds_calib_crf).cpu().detach().numpy(),
                        beta_crf=0.01,
                        nonconformity_score=method_parse[1],
                        **intvl_kwargs,
                    )
                ]
            elif method_parse[1] == "cqr":
                cp_regions += [
                    eval_cqr_multioutput(
                        ys_calib=targets_calib.cpu().detach().numpy(),
                        preds_calib=preds_calib_cqr.cpu().detach().numpy(),
                        preds_eval=preds_test_cqr.cpu().detach().numpy(),
                        sig_lvls=sig_lvls,
                    )
                ]
            else:
                raise ValueError("Invalid method")
        cp_regions = np.array(cp_regions)
        volumes = []
        coverage = []
        for jj in range(len(SPLIT_CONF_METHODS)):
            vol, cov = eval_conf_hyperrectangle_metrics(cp_regions[jj], targets_test.cpu().numpy())
            volumes += [vol]
            coverage += [cov]
    volumes = np.array(volumes)  # with shape (n_methods, len(sig_lvls))
    coverage = np.array(coverage)
    return {"volumes": volumes, "coverage": coverage, "test_loc_error": loc_error, "test_acc": acc}


def main(config):
    """Main function that launches experiment."""
    save_dir = f"results/pet_{config['backbone']}/"
    print(f"{config['backbone']}")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_mdl = None
    if config["save_model"]:
        save_dir_mdl = f"model_ckpts/pet_{config['backbone']}/"
        os.makedirs(save_dir_mdl, exist_ok=True)

    set_seed(config["seed"])  # seed

    print("device", DEVICE)

    laplace_refine = config["cp_type"] == "full_refine"

    print(f"Seed {config['seed']}")
    results = run_experiment(
        config["seed"],
        config["cp_type"],
        config["sig_lvls"],
        config["split_calib_size"],
        config["backbone"],
        laplace_refine,
        save_dir_mdl,
    )

    print()
    print(f"Test-loc-error={results['test_loc_error']:.3f},   Test-acc={results['test_acc']:.3f}")

    methods = (
        FULL_CONF_METHODS if config["cp_type"] in ["full", "full_refine"] else SPLIT_CONF_METHODS
    )
    print()
    for kk, sig_lvl in enumerate(config["sig_lvls"]):
        print(f"Coverage: {100*(1-sig_lvl):.0f}%")

        for jj, method in enumerate(methods):
            print(
                f"{method:18s}: cov={results['coverage'][jj,kk].item():.3f}  "
                + f"vol={results['volumes'][jj,kk].item():.4f}"
            )
        print()

    fname = "pet"
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
    parser.add_argument("--cp_type", default="full", choices=["full", "full_refine", "split"])
    parser.add_argument("--split_calib_size", type=float, default=0.25)
    parser.add_argument("--sig_lvls", nargs="+", type=float, default=[0.15, 0.1, 0.05])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", default="vgg19", choices=["vgg19", "resnet18", "resnet34"])
    parser.add_argument("--save_model", action="store_true")
    parser.set_defaults(save_model_state=False)
    config = parser.parse_args().__dict__
    main(config)
