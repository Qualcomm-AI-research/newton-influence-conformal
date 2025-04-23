# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Implement dataset and model for bounding box prediction tasks."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from numpy.random import default_rng
from numpy.typing import NDArray
from skimage import io
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, SmoothL1Loss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    VGG19_Weights,
    resnet18,
    resnet34,
    vgg19,
)
from torchvision.ops import box_iou
from torchvision.transforms import v2

from lib.nn_training import pinball_loss

SIZE_H_W = 224  # Size of image for both height and width.
# Normalization operation from ImageNet (and its inverse).
normalize_imagenet = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class PetDataset(Dataset):
    """Wrap the Oxford-IIT-PET dataset.

    Only uses official train split since test split does not come with bounding box annotations
    (counted 3686 annotated images overall).

    Attributes
    ----------
    has_calib_set: bool
        Whether the data set has a separate calibration dataset.
    split: str
        Which of the data splits to fetch. Either one of {'train', 'calib', 'test'}.
    device: str
        The device where the data should be mapped to.
    data_aug: bool
        Whether to perform data augmentation during training.
    with_labels: bool
        Whether to return class labels.
    pets_frame: NDArray
        The annotated bounding boxes.
    indices: NDArray
        The indices corresponding to the data points in the desired data split.
    transform_resize: Callable
        Resize the images for compatibility with ImageNet type architectures.
    transform_image: Callable
        Normalizes and map images to torch Tensors.
    transform_aug: Callable
        Applies a random horizontal flip.
    """

    def __init__(
        self,
        csv_file: str,
        split: str = "train",
        split_train_size: float = 0.8,
        split_calib_size: float = 0.0,
        data_aug: bool = False,
        with_labels: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ):
        """Initialize dataset from csv file and relative split sizes.

        Parameters
        ----------
        csv_file: str
            Path to the csv file with annotations.
        split: str
            Which of the data splits to fetch. Either one of {'train', 'calib', 'test'}.
        split_train_size: float
            Fraction of the data reserved for training and calibration. The remaining data points
            are reserved for testing.
        split_calib_size: float
            Fraction of training data to be used for calibration, i.e., not used for training the
            model, only for conformal calibration.
        data_aug: bool
            Whether to apply data augmentation.
        with_labels: bool
            Whether to return class labels.
        seed: int
            Random seed which defines the data splits.
        device: str
            The device to which the data will be mapped to.
        """
        assert split in ["train", "test", "calib"]
        self.has_calib_set = split_calib_size > 0.0
        assert not (
            not self.has_calib_set and split == "calib"
        ), "calib_size needs to be larger than 0"
        self.split = split
        self.device = device
        self.data_aug = data_aug
        self.with_labels = with_labels
        # Read bounding box annotations.
        pets_frame_all = pd.read_csv(csv_file)
        # Set numpy's random number generator.
        rng = default_rng(seed)
        # Shuffle the data and split it into train and test splits.
        n_train = int(split_train_size * len(pets_frame_all))
        perm = rng.permutation(len(pets_frame_all))
        indices_train = perm[:n_train]
        indices_test = perm[n_train:]
        # Construct calibration dataset from the training data.
        if self.has_calib_set:
            n_calib = int(split_calib_size * n_train)
            rng.shuffle(indices_train)
            indices_calib = indices_train[n_train - n_calib :]
            indices_train = indices_train[: n_train - n_calib]
            assert (len(indices_train) + len(indices_calib) + len(indices_test)) == len(
                pets_frame_all
            )
        # Get the relevant data points.
        if split == "train":
            self.pets_frame = pets_frame_all.iloc[indices_train]
            self.indices = indices_train
        elif split == "calib":
            self.pets_frame = pets_frame_all.iloc[indices_calib]
            self.indices = indices_calib
        elif split == "test":
            self.pets_frame = pets_frame_all.iloc[indices_test]
            self.indices = indices_test
        # Transforms to ensure consistent with ImageNet pretrained models (e.g. resnet).
        self.transform_resize = v2.Compose([v2.ToImage(), v2.Resize(size=(SIZE_H_W, SIZE_H_W))])
        self.transform_image = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), normalize_imagenet]
        )
        self.transform_aug = v2.RandomHorizontalFlip(
            0.5
        )  # Commonly used data augmentation for object localization.

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.pets_frame)

    def __getitem__(
        self, idx: Union[List[int], NDArray, Tensor]
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Return datapoint identified by index idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Read the image and bounding box coordinates and map them to torch tensors.
        img_path = self.pets_frame.iloc[idx, 0]
        image = io.imread(img_path)
        height, width, _ = image.shape
        bbox = np.asarray(self.pets_frame.iloc[idx, 1:-1].values, dtype=np.float32)
        bbox = tv_tensors.BoundingBoxes(
            torch.as_tensor(bbox), format="XYXY", canvas_size=(height, width)
        )
        label = torch.tensor([self.pets_frame.iloc[idx, -1]], dtype=torch.float32)
        # Apply transforms
        image, bbox = self.transform_resize(image, bbox)
        if self.data_aug:
            image, bbox = self.transform_aug(image, bbox)
        image = self.transform_image(image)
        # Rescale to [0,1].
        bbox = bbox.squeeze() / SIZE_H_W
        if self.with_labels:
            return image.to(self.device), bbox.to(self.device), label.to(self.device)
        return image.to(self.device), bbox.to(self.device)


class PetDatasetResiduals(Dataset):
    """Handle the residuals of the bounding box predictor on PetDataset.

    Dataset class accompanying Oxford-IIT-PET to use for conformalized residual
    fitting (CRF).

    Attributes
    ----------
    targets_all: Tensor
        The residuals, i.e., the difference between model prediction and ground truth.
    pets_frame: NDArray
        The annotated bounding boxes.
    transform_resize: Callable
        Resize the images for compatibility with ImageNet type architectures.
    transform_image: Callable
        Normalizes and map images to torch Tensors.
    device: str
        The device to which the data will be mapped to.
    """

    def __init__(self, csv_file: str, indices_df: Tensor, targets_all: Tensor, device: str = "cpu"):
        """Initialize the dataset class from a csv file.

        Parameters
        ----------
        csv_file: str
            Path to the csv file with annotations.
        indices_df: Tensor
            The indices corresponding to the data points in the desired data split.
        targets_all: Tensor
            The residuals, i.e., the difference between model prediction and ground truth.
        device: str
            The device to which the data will be mapped to.
        """
        self.targets_all = targets_all  # Tensor of shape (number of datapoints, number of targets)
        self.device = device
        pets_frame_all = pd.read_csv(csv_file)
        self.pets_frame = pets_frame_all.iloc[indices_df]
        self.transform_resize = v2.Compose([v2.ToImage(), v2.Resize(size=(SIZE_H_W, SIZE_H_W))])
        self.transform_image = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), normalize_imagenet]
        )

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.pets_frame)

    def __getitem__(self, idx: Union[List[int], NDArray, Tensor]) -> Tuple[Tensor, Tensor]:
        """Return datapoint identified by index idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.pets_frame.iloc[idx, 0]
        image = io.imread(img_path)
        image = self.transform_resize(image)
        image = self.transform_image(image)
        target = self.targets_all[idx]
        return image.to(self.device), target.to(self.device)


def get_backbone(backbone: str) -> Tuple[nn.Module, int]:
    """Return the backbone model and the number of learned features.

    The model is initialized to ImageNet pretrained weights and the learned features
    correspond to the input features to the last linear layer.
    """
    if backbone == "resnet18":
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        n_features = base_model.fc.in_features
        base_model.fc = nn.Identity()  # effectively remove last layer
    elif backbone == "resnet34":
        base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        n_features = base_model.fc.in_features
        base_model.fc = nn.Identity()  # effectively remove last layer
    elif backbone == "vgg19":
        base_model = vgg19(weights=VGG19_Weights.DEFAULT)
        n_features = base_model.classifier[-1].in_features
        base_model.classifier[-1] = nn.Identity()
    else:
        raise ValueError("unrecognised backbone arg")
    return base_model, n_features


class BBoxPredictor(nn.Module):
    """Define a bounding box predictor on top of a (pretrained) architecture.

    It extends the pretrained model with linear heads outputting all 4 coordinates of bounding
    boxes. It predicts the bounding box coordinates `n_quantiles` times, in case the
    model is used to estimate different quantile levels.

    Attributes
    ----------
    base_model: nn.Module
        (Pretrained model) that serves as feature extractor.
    with_clf: bool
        Whether to include a classifier head.
    n_quantiles: int
        Number of quantile levels the model will be trained to predict.
    bbox: nn.Linear
        Linear head used to predict all 4 bounding box coordinates. It outputs 4 coordinates for
        each quantile level, and thus the output dimension is equal to 4 * n_quantiles.
    clf: nn.Linear
        The classifier head. The Pets dataset only contains 2 classes, so the classifier head
        outputs a single value.
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_features: int,
        with_clf: bool = True,
        freeze_base_model: bool = False,
        n_quantiles: int = 1,
    ):
        """Initialize the model from existing pretrained backbone.

        Parameters
        ----------
        base_model: nn.Module
            (Pretrained model) that serves as feature extractor.
        n_features: int
            Number of features output by the base model.
        with_clf: bool
            Whether to include a classifier head.
        freeze_base_model: bool
            If true, the base model is kept fixed. Otherwise, the base model is updated with linear
            heads for the purpose of bounding box prediction.
        n_quantiles: int
            Number of quantile levels the model will be trained to predict.
        """
        super().__init__()
        self.base_model = base_model
        self.with_clf = with_clf
        self.n_quantiles = n_quantiles
        # Multiple heads for bbox predictor and classifier.
        self.bbox = nn.Linear(n_features, 4 * n_quantiles)
        if with_clf:
            self.clf = nn.Linear(n_features, 1)
        # Turn off gradient computation for the base model if necessary.
        for p in self.base_model.parameters():
            p.requires_grad = not freeze_base_model

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Evaluate forward pass."""
        features = self.base_model(x)
        if self.with_clf:
            return self.bbox(features).reshape((len(x), 4, self.n_quantiles)).squeeze(), self.clf(
                features
            )
        return self.bbox(features).reshape((len(x), 4, self.n_quantiles)).squeeze()


def do_evalbatch_bbox_clf(
    batch_target_bbox: Tensor,
    batch_target_lbl: Tensor,
    batch_pred_bbox: Tensor,
    batch_pred_logit: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Evaluate localization error (IOU).

    This corresponds to on the metric from Imagenet challenge (threshold
    intersection over union, or IOU) and accuracy.

    Parameters
    ----------
    batch_target_bbox: Tensor
        Ground truth bounding box coordinates.
    batch_target_lbl: Tensor
        Ground truth class labels.
    batch_pred_bbox: Tensor
        Predicted bounding box coordinates.
    batch_pred_logit: Tensor
        Predict logits for the classification task.

    Returns
    -------
    loc_error: Tensor
        The localization error (IOU)
    acc: : Tensor
        The model accuracy.
    """
    batch_target_bbox *= SIZE_H_W  # Restore box to 224x224 dims.
    batch_pred_bbox *= SIZE_H_W
    iou_all = box_iou(
        batch_target_bbox, batch_pred_bbox
    ).diag()  # Take diag as operation evaluates pairwise.
    loc_error = torch.mean((iou_all <= 0.5).float()).item()
    acc = torch.sum(
        torch.heaviside(batch_pred_logit, torch.tensor(0.0)) == batch_target_lbl
    ).item() / len(batch_target_lbl)
    return loc_error, acc


def train_bbox_net(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader] = None,
    n_epochs: int = 100,
    lr: float = 1e-1,
    with_warmup: bool = True,
    with_clf: bool = True,
    quantiles: Optional[List[float]] = None,
    eval_frequency: int = 5,
) -> nn.Module:
    """Train bounding box predictor.

    Parameters
    ----------
    model: nn.Module
        The model to be trained.
    train_loader: Dataloader
        Data loader containing all the training data except for the data reserved for validation.
    valid_loader: Dataloader
        Data loader containing the validation data.
    n_epochs: int
        Number of epochs to train the model for when optimizing the marginal likelihood.
    lr: float
        Learning rate.
    with_warmup: bool
        Whether to train the model for a few epochs before start updating the learning rate.
    with_clf: bool
        Whether to include a classifier head.
    quantiles: List[float]
        Quantile levels the model will be trained to predict.
    eval_frequency: int
        Define the frequency at which we evaluate the model on the validation data.
        We run the evaluation every `eval_frequency` epochs.

    Returns
    -------
    model: nn.Module
        The trained model.
    """
    is_quantile_reg = quantiles is not None  # Whether this is a quantile regression model.
    assert not (with_clf and is_quantile_reg)
    num_train = len(train_loader.dataset)  # Number of training data points.
    # Loss used in Fast(er) R-CNN; OK to use different loss for training and Laplace post-hoc part
    criterion_bb = SmoothL1Loss(reduction="mean")
    criterion_clf = BCEWithLogitsLoss(reduction="mean")
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # Set up the scheduler (with or without a warmup phase).
    if with_warmup:
        warmup_steps = 5 * len(train_loader)
        scheduler_warmup = LinearLR(
            optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer, T_max=n_epochs * len(train_loader) - warmup_steps, eta_min=0
        )
        scheduler = SequentialLR(
            optimizer, [scheduler_warmup, scheduler_cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(train_loader), eta_min=0)
    # Train the model.
    model.train()
    print("TRAINING MODEL")
    for epoch in tqdm.trange(1, n_epochs + 1, disable=False):
        epoch_loss = 0
        for data in train_loader:
            if with_clf:
                x, y_bbox, y_lbl = data
            else:
                x, y_bbox = data
            optimizer.zero_grad()
            if with_clf:
                # Classification task.
                f_bbox, f_logit = model(x)
                # Classification and bbox objectives weighted equally.
                loss = criterion_bb(f_bbox, y_bbox) + criterion_clf(f_logit, y_lbl)
            else:
                # Quantile regression task.
                f_bbox = model(x)
                if not is_quantile_reg:
                    loss = criterion_bb(f_bbox, y_bbox)
                else:
                    loss = 0.0
                    for kk in range(4):
                        for ii, quantile in enumerate(quantiles):
                            loss += pinball_loss(
                                f_bbox[:, kk, ii].unsqueeze(1), y_bbox[:, kk].unsqueeze(1), quantile
                            )
                    loss /= len(quantiles) * 4
            # Update model.
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() * len(y_bbox)
        # Update scheduler.
        scheduler.step()
        epoch_loss /= num_train
        # Evaluate the model.
        if valid_loader is not None:
            num_val = len(valid_loader.dataset)
            if (epoch % eval_frequency) == 0 or epoch == n_epochs:
                with torch.no_grad():
                    error = 0
                    acc = 0
                    for data in valid_loader:
                        if with_clf:
                            # Assumed here that targets are actual bbox targets and not residuals.
                            x_val, y_val_bbox, y_val_lbl = data
                            f_val_bbox, f_val_logit = model(x_val)
                            loc_error_batch, acc_batch = do_evalbatch_bbox_clf(
                                y_val_bbox, y_val_lbl, f_val_bbox, f_val_logit
                            )
                            error += loc_error_batch * len(x_val)
                            acc += acc_batch * len(x_val)
                        else:
                            x_val, y_val_bbox = data
                            f_val_bbox = model(x_val)
                            if not is_quantile_reg:
                                error += criterion_bb(f_val_bbox, y_val_bbox) * len(x_val)
                            else:
                                loss = 0.0
                                for kk in range(4):
                                    for ii, quantile in enumerate(quantiles):
                                        loss += pinball_loss(
                                            f_val_bbox[:, kk, ii].unsqueeze(1),
                                            y_val_bbox[:, kk].unsqueeze(1),
                                            quantile,
                                        )
                                loss /= len(quantiles) * 4
                                error += loss * len(x_val)
                    error /= num_val
                    acc /= num_val
                    if with_clf:
                        print(
                            f"[epoch={epoch}]: loss={epoch_loss:.3f}  "
                            f"val-error={error:.3f}  val-acc={acc:.3f}"
                        )
                    else:
                        print(f"[epoch={epoch}]: loss={epoch_loss:.3f}  val-error={error:.3f}")
    return model
