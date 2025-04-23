# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/aleximmer/heteroscedastic-nn
# Copyright (c) 2021 Alex Immer, licensed under the MIT License
# License is provided for attribution purposes only, Not a Contribution

"""Implementation of Dataset classes to handle UCI regression datasets."""

import os
from typing import Tuple

import numpy as np
import sklearn.model_selection as modsel
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset

from lib.datasets_cqr import GetDataset

UCI_DATASETS = ["boston", "concrete", "energy", "kin8nm", "power", "wine", "yacht"]
CQR_DATASETS = ["bike", "community", "protein", "facebook_1", "facebook_2"]
ALL_DATASETS = UCI_DATASETS + CQR_DATASETS


class RegressionDatasets(Dataset):
    """Wrap a regression dataset together with necessary preprocessing steps.

    Attributes
    ----------
    has_valid_set: bool
        Whether the data set has a separate validation dataset.
    has_calib_set: bool
        Whether the data set has a separate calibration dataset.
    root: str
        The root of the path where the data is saved.
    data_file: str
        The path where the data is saved.
    split: str
        Which of the data splits to fetch. Either one of {'train', 'valid', 'calib', 'test'}.
    scl: StandardScaler
        Scaler used to preprocess the data.
    m: NDArray
        The mean of each variable in the data. Used by the scaler.
    s: NDArray
        The standard deviation of each variable in the data. Used by the scaler.
    data: Tensor
        The data corresponding to the covariate variables.
    targets: Tensor
        The data corresponding to the target variable.
    """

    def __init__(
        self,
        data_set: str,
        split: str = "train",
        split_train_size: float = 0.9,
        split_valid_size: float = 0.1,
        split_calib_size: float = 0.0,
        seed: int = 6,
        shuffle: bool = True,
        root: str = "data/",
        scaling: bool = True,
        kfold_idx: int = -1,
        double: bool = False,
        device: str = "cpu",
    ):
        """Initialize dataset class and define data splits."""
        assert isinstance(seed, int), "Please provide an integer random seed"
        error_msg = "invalid UCI regression dataset"
        assert data_set in ALL_DATASETS, error_msg
        assert 0.0 <= split_train_size <= 1.0, "split_train_size does not lie between 0 and 1"
        assert 0.0 <= split_valid_size <= 1.0, "split_valid_size does not lie between 0 and 1"
        assert 0.0 <= split_calib_size <= 1.0, "split_calib_size does not lie between 0 and 1"
        assert split in ["train", "valid", "test", "calib"]
        assert (
            -1 <= kfold_idx <= 9
        ), "kfold_idx does not lie between -1 and 9"  # -1 is not KFold, otherwise K=10

        self.has_valid_set = split_valid_size > 0.0
        assert not (
            not self.has_valid_set and split == "valid"
        ), "valid_size needs to be larger than 0"
        self.has_calib_set = split_calib_size > 0.0
        assert not (
            not self.has_calib_set and split == "calib"
        ), "calib_size needs to be larger than 0"
        self.root = root
        self.split = split

        if data_set in UCI_DATASETS:
            self.data_file = os.path.join(self.root, "uci", data_set, "data.txt")
            xy_full = np.loadtxt(self.data_file)
        elif data_set in CQR_DATASETS:
            if data_set == "protein":
                x, y = GetDataset("bio", os.path.join(self.root, "cqr/"))
            else:
                x, y = GetDataset(data_set, os.path.join(self.root, "cqr/"))
            xy_full = np.hstack((x, np.expand_dims(y, -1)))
        else:
            raise ValueError("Unrecognised data_set")

        if kfold_idx == -1:  # Arbitrary train/test split
            xy_train, xy_test = modsel.train_test_split(
                xy_full, train_size=split_train_size, random_state=seed, shuffle=shuffle
            )
        else:
            # Note this doesn't use split_train_size (fixed 10 folds);
            # train/test set size could be +/- 1 different across folds.
            kf = modsel.KFold(n_splits=10, shuffle=True, random_state=seed)
            indices_train, indices_test = list(kf.split(xy_full))[kfold_idx]
            xy_train = xy_full[indices_train]
            xy_test = xy_full[indices_test]

        if self.has_calib_set:
            xy_train, xy_calib = modsel.train_test_split(
                xy_train, train_size=1 - split_calib_size, random_state=seed, shuffle=shuffle
            )
            assert (len(xy_test) + len(xy_calib) + len(xy_train)) == len(xy_full)

        if self.has_valid_set:
            xy_train, xy_valid = modsel.train_test_split(
                xy_train, train_size=1 - split_valid_size, random_state=seed, shuffle=shuffle
            )
            if self.has_calib_set:
                assert (len(xy_test) + len(xy_valid) + len(xy_train)) + len(xy_calib) == len(
                    xy_full
                )
            else:
                assert (len(xy_test) + len(xy_valid) + len(xy_train)) == len(xy_full)

        if scaling:
            self.scl = StandardScaler(copy=True)
            self.scl.fit(xy_train[:, :-1])
            xy_train[:, :-1] = self.scl.transform(xy_train[:, :-1])
            xy_test[:, :-1] = self.scl.transform(xy_test[:, :-1])
            self.m = xy_train[:, -1].mean()
            self.s = xy_train[:, -1].std()
            xy_train[:, -1] = (xy_train[:, -1] - self.m) / self.s
            xy_test[:, -1] = (xy_test[:, -1] - self.m) / self.s
            if self.has_valid_set:
                xy_valid[:, :-1] = self.scl.transform(xy_valid[:, :-1])
                xy_valid[:, -1] = (xy_valid[:, -1] - self.m) / self.s
            if self.has_calib_set:
                xy_calib[:, :-1] = self.scl.transform(xy_calib[:, :-1])
                xy_calib[:, -1] = (xy_calib[:, -1] - self.m) / self.s

        # Impossible setting: if train is false, valid needs to be false too.
        if split == "train":
            self.data = torch.from_numpy(xy_train[:, :-1]).to(device)
            self.targets = torch.from_numpy(xy_train[:, -1]).to(device)
        elif split == "valid":
            self.data = torch.from_numpy(xy_valid[:, :-1]).to(device)
            self.targets = torch.from_numpy(xy_valid[:, -1]).to(device)
        elif split == "test":
            self.data = torch.from_numpy(xy_test[:, :-1]).to(device)
            self.targets = torch.from_numpy(xy_test[:, -1]).to(device)
        elif split == "calib":
            self.data = torch.from_numpy(xy_calib[:, :-1]).to(device)
            self.targets = torch.from_numpy(xy_calib[:, -1]).to(device)

        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()
        else:
            self.data = self.data.float()
            self.targets = self.targets.float()

        if self.targets.ndim == 1:
            # Make (n_samples, 1) to comply with MSE.
            self.targets = self.targets.unsqueeze(-1)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Return data point identified by given index.

        Parameters
        ----------
        index: int
            Index of the image to be fetched.

        Returns
        -------
        Tuple: (data point, target)
            The target is index of the target class.
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return self.data.shape[0]
