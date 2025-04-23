# Approximating Full Conformal Prediction for Neural Network Regression with Gauss-Newton Influence
Dharmesh Tailor, Alvaro H.C. Correia, Eric Nalisnick and Christos Louizos. "Approximating Full Conformal Prediction for Neural Network Regression with Gauss-Newton Influence." [[ICLR2025]](https://openreview.net/forum?id=vcX0k4rGTt)

Qualcomm AI Research, Qualcomm Technologies Netherlands B.V. (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.).

## Abstract

Uncertainty quantification is an important prerequisite for the deployment of deep learning models in safety-critical areas. Yet, this hinges on the uncertainty estimates being useful to the extent the prediction intervals are well-calibrated and sharp. In the absence of inherent uncertainty estimates (e.g. pretrained models predicting only point estimates), popular approaches that operate post-hoc include Laplace’s method and split conformal prediction (split-CP). However, Laplace’s method can be miscalibrated when the model is misspecified and split-CP requires sample splitting, and thus comes at the expense of statistical efficiency. In this work, we construct prediction intervals for neural network regressors post-hoc without held-out data. This is achieved by approximating the full conformal prediction method (full-CP). Whilst full-CP nominally requires retraining the model for every test point and candidate label, we propose to train just once and locally perturb model parameters using Gauss-Newton influence to approximate the effect of retraining. Coupled with linearization of the network, we express the absolute residual nonconformity score as a piecewise linear function of the candidate label allowing for an efficient procedure that avoids the exhaustive search over the output space. On standard regression benchmarks and bounding box localization, we show the resulting prediction intervals are locally-adaptive and often tighter than those of split-CP.

## Getting started

### Environment Setup
First, let's define the path where we want the repository to be downloaded.
```
REPO_PATH=<path/to/repo>
```
Now we can clone the repository.
```
git clone git@github.com:Qualcomm-AI-research/newton-influence-conformal.git $REPO_PATH
cd $REPO_PATH
```
Next, create a virtual environment.
```
python3 -m venv env
source env/bin/activate
```
Make sure to have Python ≥3.10 (tested with Python 3.10.12) and ensure the latest version of pip (tested with 24.2).
```
pip install --upgrade --no-deps pip
```
Finally, install the required packages using pip.
 ```
pip install -r requirements.txt
```
To run the code as indicated below, the project root directory needs to be added to your PYTHONPATH.
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

### Datasets
The datasets considered in the paper can be downloaded using the script `download_datasets.sh`. The data will store in the `data/` folder. 

### Repository Structure

```
|- configs/
    |- cqr.yaml         	(config file for conformal quantile regression experiments.)
    |- pet_resnet34.yaml 	(config file for Oxford Pets experiment with ResNet34.)
    |- pet_vgg19.yaml      	(config file for Oxford Pets experiment with VGG19.)
    |- uci.yaml     		(config file for UCI regression experiments.)
|- data/
    |- prepare_pets_data.py 	(preprocessing script for bounding box prediction on Oxford Pets dataset.)
|- lib/
    |- bounding_box.py		(implementation of bounding box predictors.)
    |- crr.py			(conformal ridge regression algorithms.)
    |- datasets_cqr.py		(dataset class for handling conformal quantile regression datasets.)
    |- datasets_uci.py		(dataset class for handling UCI regression datasets.)
    |- marglik_training.py	(implementation of marginal likelihood training.)
    |- models.py		(neural regressor used for UCI regression datasets.)
    |- nn_training.py		(training routines.)
    |- trainer.py		(training routines including Laplace approximation.)
    |- utils.py			(auxiliary functions used throughout.)
|- notebooks/
    |- acp_gn_example.ipynb	(example of acp_gn usage on UCI regression datasets.)
    |- postprocess.ipynb	(notebook for collecting and analyzing results.)
|- run_pets.py			(script for reproducing Oxford Pets experiment.)
|- run_synthetic.py		(script for reproducing synthetic experiment.)
|- run_uci.py			(script for reproducing UCI experiments.)
```

## Basic Usage

For a simple example on how to use our ACP-GN method on regression tasks, see `notebooks/acp_gn_example.ipynb`.

## Reproducing Experiments
This section details how to reproduce the experiments in the paper. In each case, the command line runs the experiment for a given random seed, and the results (mainly observed coverage, average interval length) are printed to the screen. Additionally, the results are also saved in the form of a dictionary in `REPO_PATH/results/` so that results across different random seeds can be aggregated. See `notebooks/postprocess.ipynb` to see how the results are collected and postprocessed. In particular, the results in the paper were obtained with seeds in linear order (1, 2, ..., num_seeds).

### UCI Regression Tasks
The basic command for running UCI regression experiments is as follows
```
python run_uci.py --dataset <dataset_name> --cp_type <cp_type> --sig_lvls 0.1 0.05 0.01 --split_calib_size <split_calib_size> --crr_algo <crr_algo> --hyperparam_opt <hyperparam_opt> --kfold --seed <seed> 
```
The parameters are explained below.
- `dataset`: The name of the dataset we want to run. See `lib/datasets_uci.py` for a list of all datasets we consider.
- `cp_type`: The class of CP methods to run, either 'full' for Full CP, 'split' for Split CP or 'full_refine' (see Section 4 in the paper). Note that for each type, the script evaluates all variants considered in the paper. 
    - If `cp_type=split`, the script runs vanilla split CP (SCP), Conformal Quantile Regression (CQR), Conformalized Residual Fitting (CRF) and our split CP method, SCP-GN. 
    - If `cp_type=full` or `cp_type=full_refine` the script computes bayesian confidence intervals via the Laplace appromixation (`bayes` in the code, but LA in the paper) and our method, ACP-GN, with different conformal ridge regression variants, CRR_standard, CRR_studentized and CRR_deleted. We found CRR_studentized to work best among the CRR variants, and that is the one we report in the main paper (Tables 1 and 2).
- `sig_lvls`: The significance levels (or alpha) which are here set to 0.1, 0.05 and 0.01 as in the paper, but the user is free to choose any set of alpha values between [0, 1].
- `split_calib_size`: The fraction of the data to be reserved for calibration. The default value is 0.5, but we also consider `split_calib_size=0.25`.
- `crr_algo`: The conformal ridge regression to use, either one of "nouretdinov", "vovk", "vovk_mod", "burnaev". See `lib/crr.py` for the implementation of each method.
- `hyperparam_opt`: The type of hyperparameter optimization to use, either 'marglik' or 'cv'.
- `kfold`: If set, CP is ran in a K-fold cross-validation fashion. This is for training the model and running CP, not for hyperparameter optimization, which is defined by `hyperparam_opt` above.
- `seed`: The random seed for reproducibility.

We divide the UCI regression datasets into three groups according to their size. For each of those groups, we vary the hyperparameter optimization method and CRR algorithm as follows.
- *Small*: "boston", "concrete", "energy", "wine", "yacht". \
K-fold ran across 10 random seeds.
```
--kfold --hyperparam_opt marglik --crr_algo vovk_mod
```
- *Medium*: "kin8nm" and "power". \
Train-test splits across 20 random seeds.
```
--hyperparam_opt marglik --crr_algo burnaev
``` 
- *Large*: "bike", "community", "protein", 
"facebook_1", "facebook_2". \
Train-test splits across 20 random seeds.
```
--hyperparam_opt cv --crr_algo burnaev
```

### Oxford Pets dataset

Before running the Oxford Pets experiments, preprocess to compute the correct bounding box information.

```bash
python data/prepare_pets_data.py --path=<path/to/oxford-iiit-pet>
```

After that, you should have the file `pet_bb_annotations.csv` inside `data/` folder. The command to reproduce the experiments is similar to that of UCI regression tasks above. We use 20 random seeds for this experiment.

```bash
python run_pets.py --cp_type <cp_type> --split_calib_size <split_calib_size> --sig_lvls <sig_lvls> --seed <seed>
```

### Synthetic dataset
The script to run experiments with synthetic follows the same format but with a few key differences.

```bash
python run_synthetic.py --method <method> --split_calib_size <split_calib_size> --sig_lvls <sig_lvls> --n_grid <n_grid> --outliers --seed <seed>
```

Most parameters remain the same as in the UCI experiments, but there are three new ones detailed below.
- `method` defines the method to be used to compute confidence intervals. This replaces the `<cp_type>` parameter and identifies a specific method instead of a class of methods. `<method>` should be either one of 
    - 'bayes': Intervals computed via the (approximate) posterior predictive distribution. 
    - 'scp': Split conformal prediction.
    - 'fullcp': Full conformal prediction. The model is retrained for each test point.
    - 'acp': Approximate full conformal prediction of [Martinez et al., 2023](https://arxiv.org/abs/2202.01315).
    - 'acpgn': Our method, approximate full conformal prediction via Gauss-Newton influence.
- `n_grid`: The number of postulated labels to consider, which are organized in a grid. Only relevant for 'fullcp' and 'acp' methods.
- `outliers`: Whether to include outliers in the synthetic data. If this flag is set, the data will include outliers. Otherwise, the data will contain no outliers.


## Citation

If you find our work useful, please cite
```
@inproceedings{tailor2025approximating,
	title={Approximating Full Conformal Prediction for Neural Network Regression with Gauss-Newton Influence},
	author={Dharmesh Tailor and Alvaro Correia and Eric Nalisnick and Christos Louizos},
	booktitle={The Thirteenth International Conference on Learning Representations},
	year={2025},
	url={https://openreview.net/forum?id=vcX0k4rGTt}
}
```
