# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Functions implementing different Conformal Ridge Regression (CRR) variants."""

import math
import multiprocessing
import sys
from enum import Enum
from typing import List, Optional

import numpy as np
import portion as intervals
import scipy as sp
from numpy.typing import NDArray

from lib.crr_nouretdinov import eval_crr_nouretdinov

FULL_CONF_METHODS = [
    "bayes",
    "crr_standard",
    "crr_studentized",
    "crr_deleted",
]

SPLIT_CONF_METHODS = ["scp_standard", "scp_norm_var", "scp_norm_std", "scp_cqr", "scp_crf"]

CRR_ALGOS = ["nouretdinov", "vovk", "vovk_mod", "burnaev"]

JITTER = 1e-4


class SetType(str, Enum):
    """Types of intersections between sets S of postulated labels.

    See Equation 6 in the paper for a definition of these sets.
    """

    INTERVAL = "interval"
    RAYS_UNION = "rays_union"
    RAY_CLOSEDOPEN = "ray_closedopen"
    RAY_OPENCLOSED = "ray_openclosed"
    REAL = "real"
    EMPTY = "empty"


def eval_crr_vovk(
    a_n: NDArray, a_m: float, b_n: NDArray, b_m: float, sig_lvls: List[float]
) -> NDArray:
    """Run Conformalized Ridge Regression algorithm of Vovk el at.

    Run Conformalized Ridge Regression (CRR) algorithm as elaborated in Vovk et al. 'Algorithmic
    Learning in a Random World' [1st edition, p.32].
    Main differences to original algorithm are:
    1.  When ith and (N+1)th residual are equal at a *single* point, only included once in set of
        all changepoints.
    2.  Interval constructed by taking union of open intervals and singletons.
    Return prediction interval given by convex closure of union over changepoint intervals.
    Residual for "N+1" AOI (+LOO) ridge problem `A + B*y` where A = [a_n, a_m] and B = [b_n, b_m].

    Parameters
    ----------
    a_n: NDArray with shape (N,)
        Coefficient capturing influence of model predictions on the residuals on the training data.
    a_m: float
        Coefficient capturing influence of model predictions on the residuals on the test data.
    b_n: NDArray with shape (N,)
        Coefficient capturing the effect of the postulated label on residuals on the training data.
    b_m: float
        Coefficient capturing the effect of the postulated label on residuals on the test data.
    sig_lvls: List[float]
        List of significance levels to construct PIs.

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), 2)
        The computed conformal prediction interval at each significance level.
    """
    n_data = len(a_n)
    # Enforce B >= 0.
    mask = np.where(b_n < 0)[0]
    b_n[mask] *= -1
    a_n[mask] *= -1
    if b_m < 0:
        b_m *= -1
        a_m *= -1
    sets_per_datapoint, changepts = [], []
    # Excl. (N+1)th point.
    for ii in range(n_data):
        # Constructing set of changepoints.
        if b_n[ii] != b_m:
            changept_1 = (a_n[ii] - a_m) / np.clip(b_m - b_n[ii], JITTER, None)
            changept_2 = -(a_n[ii] + a_m) / np.clip(b_m + b_n[ii], JITTER, None)
            u_i, v_i = (
                (changept_1, changept_2) if changept_1 < changept_2 else (changept_2, changept_1)
            )
            changepts += [u_i, v_i]
        elif (b_n[ii] != 0) and (a_n[ii] != a_m):
            uv_i = -0.5 * (a_n[ii] + a_m) / np.clip(b_n[ii], JITTER, None)
            changepts += [uv_i]
        # Evaluating sets per datapoint (later needed to compute rank of changepoint interval).
        if b_m > b_n[ii]:
            sets_per_datapoint += [intervals.closed(u_i, v_i)]
        elif b_m < b_n[ii]:
            intvl_u = intervals.openclosed(-np.inf, u_i)
            intvl_v = intervals.closedopen(v_i, np.inf)
            sets_per_datapoint += [intvl_u.union(intvl_v)]
        elif (b_m == b_n[ii]) and (b_n[ii] > 0) and (a_m < a_n[ii]):
            sets_per_datapoint += [intervals.closedopen(uv_i, np.inf)]
        elif (b_m == b_n[ii]) and (b_n[ii] > 0) and (a_m > a_n[ii]):
            sets_per_datapoint += [intervals.openclosed(-np.inf, uv_i)]
        elif (b_m == b_n[ii]) and (b_n[ii] == 0) and (abs(a_m) <= abs(a_n[ii])):
            sets_per_datapoint += [intervals.open(-np.inf, np.inf)]
        elif (b_m == b_n[ii]) and (b_n[ii] == 0) and (abs(a_m) > abs(a_n[ii])):
            sets_per_datapoint += [intervals.empty()]
        elif (b_m == b_n[ii]) and (a_n[ii] == a_m):
            sets_per_datapoint += [intervals.open(-np.inf, np.inf)]
        else:
            raise ValueError(
                "All possibilities for sets corresponding to each datapoint should be handled!"
            )
    sets_per_datapoint += [
        intervals.open(-np.inf, np.inf)
    ]  # Explicitly handle set (real line) for N+1 entry.
    changepts_sorted = np.sort([-np.inf] + changepts + [np.inf])
    n_changepts = len(changepts_sorted)
    # Version from Vovk textbook 1st edition that takes union of open intervals and singletons.
    # Seems to be twice as slow as original algo.
    cp_intervals = []
    for sig_lvl in sig_lvls:
        cp_interval = intervals.empty()
        for jj in range(n_changepts - 1):
            # Compute rank of each changepoint open interval.
            n_pvalue_j = 0
            intvl_j = intervals.open(changepts_sorted[jj], changepts_sorted[jj + 1])
            for _, set_dp in enumerate(sets_per_datapoint):
                n_pvalue_j += intvl_j in set_dp
            # Trivially convert rank to p-value of changepoint interval.
            pvalue_j = n_pvalue_j / (n_data + 1)
            if pvalue_j > sig_lvl:
                cp_interval = cp_interval.union(intvl_j)
        for _, changepoint in enumerate(
            changepts_sorted[1 : n_changepts - 1]
        ):  # Excluding {-inf,inf} endpoints.
            # Compute rank of each changepoint.
            n_pvalue_j = 0
            for _, set_dp in enumerate(sets_per_datapoint):
                n_pvalue_j += changepoint in set_dp
            pvalue_j = n_pvalue_j / (n_data + 1)
            if pvalue_j > sig_lvl:
                cp_interval = cp_interval.union(intervals.singleton(changepoint))
        cp_intervals.append(
            [cp_interval.lower.item(), cp_interval.upper.item()]
        )  # Only return "convex closure".
    return np.array(cp_intervals)


def eval_crr_vovk_mod(
    a_n: NDArray, a_m: float, b_n: NDArray, b_m: float, sig_lvls: List[float]
) -> NDArray:
    """Run the modified Conformalized Ridge Regression algorithm of Vovk et al.

    Run the modified Conformalized Ridge Regression (CRR) algorithm elaborated in
    Vovk et al. 'Algorithmic Learning in a Random World' [1st edition, p.33-34]. O(N)
    improvement when evaluating rank of changepoint intervals. Return prediction
    interval given by convex closure of union over changepoint intervals Residual for
    "N+1" AOI (+LOO) ridge problem `A + B*y` where A = [a_n, a_m] and B = [b_n, b_m]

    Parameters
    ----------
    a_n: NDArray with shape (N,)
        Coefficient capturing influence of model predictions on the residuals on the training data.
    a_m: float
        Coefficient capturing influence of model predictions on the residuals on the test data.
    b_n: NDArray with shape (N,)
        Coefficient capturing the effect of the postulated label on residuals on the training data.
    b_m: float
        Coefficient capturing the effect of the postulated label on residuals on the test data.
    sig_lvls: List[float]
        List of significance levels to construct PIs.

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), 2)
        The computed conformal prediction interval at each significance level.
    """
    n_data = len(a_n)
    # Enforce B >= 0.
    mask = np.where(b_n < 0)[0]
    b_n[mask] *= -1
    a_n[mask] *= -1
    if b_m < 0:
        b_m *= -1
        a_m *= -1
    changepts_per_datapoint = [()] * (n_data + 1)
    set_type_per_datapoint = [None] * (n_data + 1)
    # Excl. (N+1)th point.
    for ii in range(n_data):
        # Evaluating changepoints.
        if b_n[ii] != b_m:
            changept_1 = (a_n[ii] - a_m) / np.clip(b_m - b_n[ii], JITTER, None)
            changept_2 = -(a_n[ii] + a_m) / np.clip(b_m + b_n[ii], JITTER, None)
            u_i, v_i = (
                (changept_1, changept_2) if changept_1 < changept_2 else (changept_2, changept_1)
            )
        elif (b_n[ii] != 0) and (a_n[ii] != a_m):
            uv_i = -0.5 * (a_n[ii] + a_m) / np.clip(b_n[ii], JITTER, None)
        # Set types (and corresponding changepoints).
        # Later needed to compute rank of changepoint interval.
        if b_m > b_n[ii]:
            changepts_per_datapoint[ii] = (u_i.item(), v_i.item())
            set_type_per_datapoint[ii] = SetType.INTERVAL
        elif b_m < b_n[ii]:
            changepts_per_datapoint[ii] = (u_i.item(), v_i.item())
            set_type_per_datapoint[ii] = SetType.RAYS_UNION
        elif (b_m == b_n[ii]) and (b_n[ii] > 0) and (a_m < a_n[ii]):
            changepts_per_datapoint[ii] = (uv_i.item(),)
            set_type_per_datapoint[ii] = SetType.RAY_CLOSEDOPEN
        elif (b_m == b_n[ii]) and (b_n[ii] > 0) and (a_m > a_n[ii]):
            changepts_per_datapoint[ii] = (uv_i.item(),)
            set_type_per_datapoint[ii] = SetType.RAY_OPENCLOSED
        elif (b_m == b_n[ii]) and (b_n[ii] == 0) and (abs(a_m) <= abs(a_n[ii])):
            set_type_per_datapoint[ii] = SetType.REAL
        elif (b_m == b_n[ii]) and (b_n[ii] == 0) and (abs(a_m) > abs(a_n[ii])):
            set_type_per_datapoint[ii] = SetType.EMPTY
        elif (b_m == b_n[ii]) and (a_n[ii] == a_m):
            set_type_per_datapoint[ii] = SetType.REAL
        else:
            raise ValueError(
                "All possibilities for sets corresponding to each datapoint should be handled!"
            )
    # Explicitly handle set (real line) for N+1 entry.
    set_type_per_datapoint[n_data] = SetType.REAL
    changepts = list(sum(changepts_per_datapoint, ()))
    n_changepts = len(changepts)
    changepts_sorted = np.sort([-np.inf] + changepts + [np.inf])
    # Trick to evaluate interval/singleton ranks in O(N);
    # instead compute diff of successive ranks ("prime").
    n_prime_j_lst = [0] * (n_changepts + 1)
    m_prime_j_lst = [0] * (n_changepts + 1)  # Keep same indexing as previous.
    for set_type, changepts_dp in zip(set_type_per_datapoint, changepts_per_datapoint):
        # NB: O(NlogN) algo has if-statement for singleton but this is a special case of closed
        # interval, so should be handled automatically.
        # Assuming changepoints are unique so can be indexed but might not be generally the case.
        if set_type == SetType.EMPTY:
            pass
        elif set_type == SetType.INTERVAL:
            j1 = np.searchsorted(changepts_sorted, changepts_dp[0])
            j2 = np.searchsorted(changepts_sorted, changepts_dp[1])
            m_prime_j_lst[j1] += 1
            if j2 < n_changepts:
                m_prime_j_lst[j2 + 1] -= 1
            n_prime_j_lst[j1] += 1
            n_prime_j_lst[j2] -= 1
        elif set_type == SetType.RAY_OPENCLOSED:
            j1 = np.searchsorted(changepts_sorted, changepts_dp[0])
            m_prime_j_lst[1] += 1
            if j1 < n_changepts:
                m_prime_j_lst[j1 + 1] -= 1
            n_prime_j_lst[0] += 1
            n_prime_j_lst[j1] -= 1
        elif set_type == SetType.RAY_CLOSEDOPEN:
            j1 = np.searchsorted(changepts_sorted, changepts_dp[0])
            m_prime_j_lst[j1] += 1
            n_prime_j_lst[j1] += 1
        elif set_type == SetType.RAYS_UNION:
            j1 = np.searchsorted(changepts_sorted, changepts_dp[0])
            j2 = np.searchsorted(changepts_sorted, changepts_dp[1])
            m_prime_j_lst[1] += 1
            if j1 < n_changepts:
                m_prime_j_lst[j1 + 1] -= 1
            m_prime_j_lst[j2] += 1
            n_prime_j_lst[0] += 1
            n_prime_j_lst[j1] -= 1
            n_prime_j_lst[j2] += 1
        elif set_type == SetType.REAL:
            m_prime_j_lst[1] += 1
            n_prime_j_lst[0] += 1
        else:
            raise ValueError("Unexpected set type")
    # Evaluate interval/singleton ranks from successive differences above.
    n_rank_j_lst = [0] * (n_changepts + 1)  # Store rank of open intervals.
    m_rank_j_lst = [0] * (
        n_changepts + 1
    )  # Store rank of (finite) changepoints but keep same indexing as previous.
    n_rank_j_lst[0] = n_prime_j_lst[0]
    for jj in range(1, n_changepts + 1):
        m_rank_j_lst[jj] = m_prime_j_lst[jj] + m_rank_j_lst[jj - 1]
        n_rank_j_lst[jj] = n_prime_j_lst[jj] + n_rank_j_lst[jj - 1]
    cp_intervals = []
    for sig_lvl in sig_lvls:
        cp_interval = intervals.empty()
        # Extract open intervals with minimum rank.
        for jj in range(n_changepts + 1):
            intvl_j = intervals.open(changepts_sorted[jj], changepts_sorted[jj + 1])
            # Trivially convert rank to p-value of changepoint interval.
            pvalue_j = n_rank_j_lst[jj] / (n_data + 1)
            if pvalue_j > sig_lvl:
                cp_interval = cp_interval.union(intvl_j)
        # Extract singletons with minimum rank.
        for jj in range(1, n_changepts + 1):
            pvalue_j = m_rank_j_lst[jj] / (n_data + 1)
            if pvalue_j > sig_lvl:
                cp_interval = cp_interval.union(intervals.singleton(changepts_sorted[jj]))
        cp_intervals.append(
            [cp_interval.lower.item(), cp_interval.upper.item()]
        )  # Only return "convex closure".
    return np.array(cp_intervals)


def eval_crr_burnaev(
    a_n: NDArray, a_m: NDArray, b_n: NDArray, b_m: NDArray, sig_lvls: List[float]
) -> NDArray:
    """Run the Conformalized Ridge Regression of Burnaev and Vovk.

    CRR algorithm proposed in [Burnaev & Vovk, 2014] that takes intersection of lower
    and upper CRR prediction sets. Returns PI for all evaluation ("N+1") points in
    parallel (fully numpy ops) hence much faster than pre- Burnaev implementations.
    However, PIs are typically less tight. Will give error for small N when sig_lvl too
    small. Residual for "N+1" AOI (+LOO) ridge problem `A + B*y` where A = [a_n, a_m]
    and B = [b_n, b_m].

    Parameters
    ----------
    a_n: NDArray with shape (M, N)
        Coefficient capturing influence of model predictions on the residuals on the training data.
    a_m: NDArray with shape (M,)
        Coefficient capturing influence of model predictions on the residuals on the test data.
    b_n: NDArray with shape (N,)
        Coefficient capturing the effect of the postulated label on residuals on the training data.
    b_m: NDArray with shape (M,)
        Coefficient capturing the effect of the postulated label on residuals on the test data.
    sig_lvls: List[float]
        List of significance levels to construct PIs.

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), M, 2)
        The computed conformal prediction intervals for each data point and significance level.
    """
    n_data = a_n.shape[1]
    an_minus_am = a_n - a_m.reshape(-1, 1)
    bm_minus_bn = b_m.reshape(-1, 1) - b_n
    changepoints_lower = np.zeros_like(a_n)  # Lower
    changepoints_lower[bm_minus_bn <= 0] = -np.inf
    changepoints_lower[bm_minus_bn > 0] = an_minus_am[bm_minus_bn > 0] / np.clip(
        bm_minus_bn[bm_minus_bn > 0], JITTER, None
    )
    changepoints_upper = np.zeros_like(a_n)  # Upper
    changepoints_upper[bm_minus_bn <= 0] = np.inf
    changepoints_upper[bm_minus_bn > 0] = an_minus_am[bm_minus_bn > 0] / np.clip(
        bm_minus_bn[bm_minus_bn > 0], JITTER, None
    )
    # NB: should be possible to avoid sorting twice since arrays are almost identical.
    changepoints_lower = np.sort(changepoints_lower, axis=-1)
    changepoints_upper = np.sort(changepoints_upper, axis=-1)
    # Get lower & upper order statistic given sig. level.
    sig_lvls = np.array(sig_lvls)
    idx_lower_lst = np.floor((n_data + 1) * sig_lvls / 2.0).astype(int) - 1
    idx_upper_lst = np.ceil((n_data + 1) * (1 - sig_lvls / 2.0)).astype(int) - 1

    # If a sig_lvl is too small, set the lower/upper endpoint to -inf/inf appropriately.
    indices_lower_invalid = np.where(idx_lower_lst < 0)[0]
    idx_lower_lst[indices_lower_invalid] = 0
    indices_upper_invalid = np.where(idx_upper_lst > n_data - 1)[0]
    idx_upper_lst[indices_upper_invalid] = n_data - 1
    cp_intervals = np.concatenate(
        (
            np.expand_dims(changepoints_lower[:, idx_lower_lst], 0),
            np.expand_dims(changepoints_upper[:, idx_upper_lst], 0),
        )
    )
    cp_intervals = np.swapaxes(cp_intervals, 0, 2)
    cp_intervals[indices_lower_invalid, :, 0] = -np.inf
    cp_intervals[indices_upper_invalid, :, 1] = np.inf
    return cp_intervals


def eval_crr(
    ys: NDArray,
    preds: NDArray,
    preds_eval: NDArray,
    h_m: NDArray,
    h_mn: NDArray,
    h_n: NDArray,
    sigma_sq: float = 1.0,
    sig_lvls: Optional[List[float]] = None,
    nonconformity_score: str = "standard",
    algo: str = "burnaev",
    n_processes: Optional[int] = None,
) -> NDArray:
    """Evaluate one of the possible Conformalized Ridge Regression algorithms.

    Supports different implementations that evaluates PI for all
    test points ("N+1").

    Parameters
    ----------
    ys: NDArray with shape (N,)
        Train targets array.
    preds: NDArray with shape (N,)
        Predictions on train set.
    preds_eval: NDArray with shape (M,)
        Predictions on test/grid points.
    h_m: NDArray with shape (M,)
        Leverage/marginal variance on test/grid points.
    h_mn: NDArray with shape (M, N)
        Cross-leverages/variances between test and train points.
    h_n: NDArray with shape (N,)
        Leverage/marginal variance on train set.
    sigma_sq: float
        Observation noise (variance).
    sig_lvls: List[float]
        List of significance levels to construct PI (default=0.05).
    nonconformity_score: str
        Choice of "standard", "deleted" (i.e. jackknife) or "studentized".
    algo: str
        CRR implementation choice of "nouretdinov", "vovk", "vovk_mod", "burnaev".
    n_processes: int
        Number of worker processes for multiprocessing when algo != "burnaev".

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), M, 2)
        The computed conformal prediction interval for each data point and significance level.
    """
    if sig_lvls is None:
        sig_lvls = [0.05]
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1 or 1
    n_eval = len(preds_eval)
    h_m = h_m * (1 / sigma_sq)
    h_mn = h_mn * (1 / sigma_sq)
    h_n = h_n * (1 / sigma_sq)
    b_n = -h_mn / (1 + h_m.reshape(-1, 1))  # (M,N)
    b_m = 1 / (1 + h_m)  # (M,)
    a_n = (ys - preds).reshape(1, -1) + (
        (h_mn / (1 + h_m.reshape(-1, 1))) * preds_eval.reshape(-1, 1)
    )  # (M,N)
    a_m = -preds_eval / (1 + h_m)  # (M,)
    if nonconformity_score in ["deleted", "studentized"]:
        h_m_bar = h_m / (1 + h_m)  # (M,)
        h_n_bar = h_n.reshape(1, -1) - h_mn**2 / (1 + h_m.reshape(-1, 1))  # (M,N)
        norm_m = 1 - h_m_bar
        norm_n = 1 - h_n_bar
        norm_m = np.clip(norm_m, JITTER, 1 - JITTER)
        norm_n = np.clip(norm_n, JITTER, 1 - JITTER)
        if nonconformity_score == "studentized":
            norm_m = np.sqrt(norm_m)
            norm_n = np.sqrt(norm_n)
        b_n = b_n / norm_n
        b_m = b_m / norm_m
        a_n = a_n / norm_n
        a_m = a_m / norm_m
    if algo in ["nouretdinov", "vovk", "vovk_mod"]:
        eval_crr_per_eval_pt = getattr(sys.modules[__name__], "eval_crr_" + algo)
        pool = multiprocessing.Pool(n_processes)
        crr_args_all = [
            (a_n[jj, :], a_m[jj].item(), b_n[jj, :], b_m[jj].item(), sig_lvls)
            for jj in range(n_eval)
        ]
        out = pool.starmap(eval_crr_per_eval_pt, crr_args_all)
        pool.close()
        pool.join()
        cp_intervals = np.swapaxes(np.array(out), 0, 1)
    elif algo == "burnaev":
        cp_intervals = eval_crr_burnaev(a_n, a_m, b_n, b_m, sig_lvls)
    else:
        raise ValueError("Invalid algo argument")
    return cp_intervals


def eval_acp(
    ys: NDArray,
    preds: NDArray,
    preds_eval: NDArray,
    h_m: NDArray,
    h_mn: NDArray,
    h_n: NDArray,
    sigma_sq: float = 1.0,
    sig_lvls: Optional[List[float]] = None,
    n_grid: int = 200,
) -> NDArray:
    """Run Approximate Conformal Prediction with Gauss-Newton Influence (ACP-GN).

    The Hessian is approximated by Gauss-Newton and with AOI nonconformity score.

    Parameters
    ----------
    ys: NDArray with shape (N,)
        Ground truth labels for the training data.
    preds: NDArray with shape (N,)
        Model predictions on the training data.
    preds_eval: NDArray with shape (M,)
        Model predictions on the test data.
    h_m: NDArray with shape (M,)
        Leverage/marginal variance on the M test points.
    h_mn: NDArray with shape (M, N)
        Leverage/marginal variance on train and test points.
    h_n: NDArray with shape (N,)
        Leverage/marginal variance on the N training points.
    sigma_sq : float
        Observation noise (variance).
    sig_lvls: List[float]
        List of significance levels to construct PI (default=0.05).
    n_grid: int
        Number of values in the grid used to define the postulated labels.

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), M, 2)
        The confidence intervals for each test data point and significance level.
    """
    if sig_lvls is None:
        sig_lvls = [0.05]
    sig_lvls = np.array(sig_lvls)
    n_train = len(ys)  # equal to N.
    n_test = len(preds_eval)  # equal to M.
    y_grid = np.linspace(ys.min(), ys.max(), n_grid)
    conf_threshold = np.ceil((n_train + 1) * (1 - sig_lvls)).astype(int)
    # Normalize the leverage by the observation noise.
    h_m = h_m * (1 / sigma_sq)
    h_mn = h_mn * (1 / sigma_sq)
    h_n = h_n * (1 / sigma_sq)
    # Scores on train points shp (M, n_grid, N).
    scores_train = np.abs(
        (ys - preds)[None, None, :]
        - (y_grid[None, :] - preds_eval[:, None])[:, :, None] * np.expand_dims(h_mn, 1)
    )
    # Scores on test points shp (M, n_grid).
    scores_test = np.abs(
        y_grid[None, :]
        - preds_eval[:, None]
        - (y_grid[None, :] - preds_eval[:, None]) * h_m[:, None]
    )  # (M, n_grid).
    rank = np.sum(scores_train <= scores_test[:, :, None], axis=-1) + 1  # with shape (M, n_grid).
    # Define the grid mask.
    grid_mask = np.expand_dims(rank, 0) <= np.expand_dims(
        conf_threshold, [1, 2]
    )  # with shape (n_sig_lvl, M, n_grid).
    cp_intervals = [[] for _ in range(len(sig_lvls))]
    for idx_siglvl in range(len(sig_lvls)):
        for idx_testpoint in range(n_test):
            mask = grid_mask[idx_siglvl, idx_testpoint]  # with shape (n_grid,).
            cp_set = y_grid[mask]
            cp_intervals[idx_siglvl].append([cp_set.min(), cp_set.max()])
    cp_intervals = np.array(cp_intervals)  # with shape (len(sig_lvls), M, 2).
    return cp_intervals


def eval_jackknife(
    ys: NDArray,
    preds: NDArray,
    preds_eval: NDArray,
    h_mn: NDArray,
    h_n: NDArray,
    sigma_sq: float = 1.0,
    sig_lvls: Optional[List[float]] = None,
    method: str = "standard",
) -> NDArray:
    """Run jackknife & jackknife+ [Barber et. al., 2020].

    Parameters
    ----------
    ys: NDArray with shape (N,)
        Ground truth labels for the training data.
    preds: NDArray with shape (N,)
        Model predictions on the training data.
    preds_eval: NDArray with shape (M,)
        Model predictions on the test data.
    h_mn: NDArray with shape (M, N)
        Leverage/marginal variance on train and test points.
    h_n: NDArray with shape (N,)
        Leverage/marginal variance on the N training points.
    sigma_sq : float
        Observation noise (variance).
    sig_lvls: List[float]
        List of significance levels to construct PI (default=0.05).
    method: str
        Either `standard` for original jackknife or `plus` for jackknife+.

    Returns
    -------
    jackknife_intervals: NDArray with shape (len(sig_lvls), M, 2)
        The confidence intervals for each test data point and significance level.
    """
    n_data = len(ys)
    if sig_lvls is None:
        sig_lvls = [0.05]
    sig_lvls = np.array(sig_lvls)

    h_mn = h_mn * (1 / sigma_sq)
    h_n = h_n * (1 / sigma_sq)
    h_n = np.clip(h_n, JITTER, 1 - JITTER)

    loo_residuals_abs = np.abs((ys - preds) / (1 - h_n))
    idx_upper_lst = np.ceil((n_data + 1) * (1 - sig_lvls)).astype(int) - 1

    if method == "standard":
        loo_residuals_abs = np.sort(loo_residuals_abs)
        quantiles_all = loo_residuals_abs[idx_upper_lst]
        quantiles_all = np.expand_dims(quantiles_all, -1)

        jackknife_intervals = np.concatenate(
            (
                np.expand_dims(np.expand_dims(preds_eval, 0) - quantiles_all, 0),
                np.expand_dims(np.expand_dims(preds_eval, 0) + quantiles_all, 0),
            )
        )
        jackknife_intervals = np.permute_dims(jackknife_intervals, axes=(1, 2, 0))
    elif method == "plus":
        preds_eval_loo = preds_eval.reshape(-1, 1) - (ys - preds).reshape(
            1, -1
        ) * h_mn / h_n.reshape(
            1, -1
        )  # (M,N)
        loo_residuals_abs = loo_residuals_abs.reshape(1, -1)
        idx_lower_lst = np.floor((n_data + 1) * sig_lvls).astype(int) - 1

        vals_lower = preds_eval_loo - loo_residuals_abs
        vals_lower = np.sort(vals_lower, axis=-1)
        vals_upper = preds_eval_loo + loo_residuals_abs
        vals_upper = np.sort(vals_upper, axis=-1)

        jackknife_intervals = np.concatenate(
            (
                np.expand_dims(vals_lower[:, idx_lower_lst], 0),
                np.expand_dims(vals_upper[:, idx_upper_lst], 0),
            )
        )
        jackknife_intervals = np.swapaxes(jackknife_intervals, 0, 2)
    else:
        raise ValueError("method can only be standard or plus")

    return jackknife_intervals


def eval_bayes(
    fmu: NDArray, fvar: NDArray, sigma_sq: float = 1.0, sig_lvls: Optional[List[float]] = None
) -> NDArray:
    """Evaluate Bayes confidence intervals in the single-output setting.

    Parameters
    ----------
    fmu: NDArray with shape (M,)
        Mean of the target variable.
    fvar: NDArray with shape (M,)
        Variance of target variable.
    sigma_sq : float
        Observation noise (variance).
    sig_lvls: List[float]
        List of significance levels to construct PI.

    Returns
    -------
    bayes_interval: NDArray with shape (M, len(sig_lvls), 2)
        The confidence interval for each data point and confidence level.
    """
    if sig_lvls is None:
        sig_lvls = [0.05]
    quantiles = -sp.stats.norm().ppf(np.array(sig_lvls) / 2)
    y_test_std = np.sqrt(fvar + sigma_sq)
    bayes_interval = np.concatenate(
        (
            np.expand_dims(
                fmu.reshape(1, -1) - y_test_std.reshape(1, -1) * quantiles.reshape(-1, 1), -1
            ),
            np.expand_dims(
                fmu.reshape(1, -1) + y_test_std.reshape(1, -1) * quantiles.reshape(-1, 1), -1
            ),
        ),
        axis=-1,
    )
    return bayes_interval


def eval_split_cp(
    ys_calib: NDArray,
    preds_calib: NDArray,
    preds_eval: NDArray,
    h_eval: Optional[NDArray] = None,
    h_calib: Optional[NDArray] = None,
    sigma_sq: float = 1.0,
    norm_eval: Optional[NDArray] = None,
    norm_calib: Optional[NDArray] = None,
    beta_crf: float = 1.0,
    sig_lvls: Optional[List[float]] = None,
    nonconformity_score: str = "standard",
) -> NDArray:
    """Run normalized split CP variety.

    This is scaled by std/var of Laplace posterior predictive.
    Related to standard/studentized residual variety of full-CP.

    Parameters
    ----------
    ys_calib: NDArray with shape (N,)
        Ground truth labels for the training data.
    preds_calib: NDArray with shape (N,)
        Model predictions on the training data.
    preds_eval: NDArray with shape (M,)
        Model predictions on the test data.
    h_eval: NDArray with shape (M,)
        Leverage/marginal variance on the M test points.
    h_calib: NDArray with shape (MN,)
        Leverage/marginal variance on train and test points.
    sigma_sq : float
        Observation noise (variance).
    norm_eval : NDArray with shape (M,)
        Normalization on test set given by CRF (conformal residual fitting).
    norm_calib : NDArray with shape (N,)
        Normalization on calibration set given by CRF (conformal residual fitting).
    beta_crf : float
        Offset used in NCP e.g. see [Papadoupolous & Haramlambous, 2011].
    sig_lvls: List[float]
        List of significance levels to construct PI (default=0.05).
    nonconformity_score: str
        Choice of "standard", "norm_var", "norm_std", "crf".

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), M, 2)
        The confidence intervals for each test data point and significance level.
    """
    if nonconformity_score in ["norm_var", "norm_std"]:
        assert (
            h_eval is not None and h_calib is not None
        ), "h_eval, h_calib needed for normalized nonconformity score"
    n_calib = len(ys_calib)
    if sig_lvls is None:
        sig_lvls = [0.05]
    sig_lvls = np.array(sig_lvls)

    scores = np.abs(ys_calib - preds_calib)

    if nonconformity_score in ["norm_var", "norm_std"]:
        y_var_calib = h_calib + sigma_sq
        y_var_calib = np.clip(y_var_calib, JITTER, None)
        normalization = y_var_calib if nonconformity_score == "norm_var" else np.sqrt(y_var_calib)
        scores = scores / normalization
    elif nonconformity_score == "crf":
        normalization = norm_calib + beta_crf
        scores = scores / normalization

    scores = np.sort(scores)
    idx_upper_lst = np.ceil((n_calib + 1) * (1 - sig_lvls)).astype(int) - 1
    quantiles_all = np.ones_like(sig_lvls) * np.inf
    quantiles_all[idx_upper_lst < n_calib] = scores[idx_upper_lst[idx_upper_lst < n_calib]]
    quantiles_all = np.expand_dims(quantiles_all, -1)

    if nonconformity_score in ["norm_var", "norm_std"]:
        y_var_eval = h_eval + sigma_sq
        y_var_eval = np.clip(
            y_var_eval, JITTER, None
        )  # only clipping to be consistent with score normalization above
        normalization = y_var_eval if nonconformity_score == "norm_var" else np.sqrt(y_var_eval)
    elif nonconformity_score == "crf":
        normalization = norm_eval + beta_crf
    else:
        normalization = np.ones_like(preds_eval)

    cp_intervals = np.concatenate(
        (
            np.expand_dims(
                np.expand_dims(preds_eval, 0) - np.expand_dims(normalization, 0) * quantiles_all, 0
            ),
            np.expand_dims(
                np.expand_dims(preds_eval, 0) + np.expand_dims(normalization, 0) * quantiles_all, 0
            ),
        )
    )
    cp_intervals = np.moveaxis(cp_intervals, (0, 1, 2), (2, 0, 1))
    return cp_intervals


def eval_cqr(
    ys_calib: NDArray, preds_calib: NDArray, preds_eval: NDArray, sig_lvls: List[float]
) -> NDArray:
    """Run Conformal Quantile Regression (CQR).

    Parameters
    ----------
    ys_calib: NDArray with shape (N,)
        Calibration targets array.
    preds_calib: NDArray with shape (N, len(sig_lvls), 2)
        Predictions on calibration set.
    preds_eval: NDArray with shape (M, len(sig_lvls), 2)
        Predictions on test points.
    sig_lvls: List[float]
        List of significance levels to construct PI.

    Returns
    -------
    cp_intervals: NDArray with shape (len(sig_lvls), M, 2)
        The confidence intervals for each test data point and significance level.
    """
    n_calib = len(ys_calib)
    sig_lvls = np.array(sig_lvls)
    scores_all = preds_calib - ys_calib[:, None, None]
    scores_all[:, :, 1] *= -1
    scores_all = np.max(scores_all, axis=-1)  # (N, len(sig_lvls))
    quantiles_all = []
    for ii, sig_lvl in enumerate(sig_lvls):
        scores = scores_all[:, ii]  # (N,)
        scores = np.sort(scores)
        idx_upper = np.ceil((n_calib + 1) * (1 - sig_lvl)).astype(int) - 1
        if idx_upper < n_calib:
            quantiles_all.append(scores[idx_upper])
        else:
            quantiles_all.append(np.inf)
    quantiles_all = np.array(quantiles_all)
    cp_intervals = (
        preds_eval + np.vstack((-quantiles_all, quantiles_all)).T[None, :]
    )  # (M, len(sig_lvls), 2)
    cp_intervals = np.moveaxis(cp_intervals, (0, 1, 2), (1, 0, 2))
    return cp_intervals


def eval_bayes_multioutput(
    fmu: NDArray,
    fcov: NDArray,
    sigma_sq: float,
    sig_lvls: Optional[List[float]] = None,
    ys: Optional[NDArray] = None,
    scale: Optional[NDArray] = None,
) -> NDArray:
    """Evaluate Bayes confidence intervals in the multi-output setting.

    Parameters
    ----------
    fmu: NDArray with shape (M, n_output)
        Mean of each target variable.
    fcov: NDArray with shape (M, n_output, n_output)
        Covariance matrix.
    sigma_sq : float
        Observation noise (variance).
    sig_lvls: List[float]
        List of significance levels to construct PI.
    ys: NDArray with shape (M, n_output)
        Ground truth labels.
    scale: NDArray with shape (n_outputs,)
        Reverse transform of standardization for hypervolume evaluation.

    Returns
    -------
    conf_region_vol_avg: NDAarray
        The average volume of the confidence region at each significance level.
    coverage: NDArray with shape (len(sig_lvls),)
        The average coverage at each significance level.
    """
    if sig_lvls is None:
        sig_lvls = [0.05]
    n_outputs = fmu.shape[-1]
    if scale is None:
        scale = np.ones(n_outputs)
    if isinstance(scale, (int, float)):
        scale = scale * np.ones(n_outputs)
    # Quantile function for chi-squared distribution with `n_outputs` degrees of freedom.
    quantiles = sp.stats.chi2(df=n_outputs).ppf(1 - np.array(sig_lvls))
    y_post_cov = fcov + sigma_sq * np.diag(np.ones(n_outputs))[None, :]
    y_post_cov_scaled = (
        scale[None, :, None] * y_post_cov * scale[None, None, :]
    )  # Reverse transform of standardization; ensure volume calc in original scale.
    # Evaluate volume.
    vol_nball = math.pi ** (n_outputs / 2) / math.gamma(
        n_outputs / 2 + 1
    )  # Volume of k-dim unit ball.
    conf_region_vols = (
        (quantiles ** (n_outputs / 2))[:, None]
        * np.sqrt(np.array([np.linalg.det(cov).item() for cov in y_post_cov_scaled]))[None, :]
        * vol_nball
    )  # (len(sig_lvls), M)
    conf_region_vol_avg = np.mean(conf_region_vols, axis=-1)  # (len(sig_lvls),)

    if ys is not None:
        cov_inv_all = np.array([np.linalg.inv(cov) for cov in y_post_cov])
        cov_check = np.einsum("mk,mkc,mc->m", ys - fmu, cov_inv_all, ys - fmu)  # (M,)
        coverage = np.mean(cov_check[None, :] <= quantiles[:, None], axis=-1)  # (len(sig_lvls),)
        return conf_region_vol_avg, coverage
    else:
        return conf_region_vol_avg


def eval_split_cp_multioutput(
    ys_calib: NDArray,
    preds_calib: NDArray,
    preds_eval: NDArray,
    h_eval: Optional[NDArray] = None,
    h_calib: Optional[NDArray] = None,
    sigma_sq: float = 1.0,
    norm_eval: Optional[NDArray] = None,
    norm_calib: Optional[NDArray] = None,
    beta_crf: float = 1.0,
    sig_lvls: Optional[NDArray] = None,
    nonconformity_score: str = "standard",
    with_bonferroni: bool = True,
) -> NDArray:
    """Run split CP on vector regression. Strictly only need diagonal of leverages.

    Parameters (N=n_calib, M=n_test, K=n_outputs).
    ----------
    ys_calib: NDArray with shape (N, K)
        Calibation targets array.
    preds_calib: NDArray with shape (N, K)
        Predictions on calibration set.
    preds_eval: NDArray with shape (M, K)
        Predictions on test/grid points.
    h_eval: NDArray with shape (M, K, K)
        Leverage/marginal variance on test/grid points.
    h_calib: NDArray with shape (N, K, K)
        Leverage/marginal variance on train set.
    sigma_sq: float
        Observation noise (variance).
    norm_eval : NDArray with shape (M, K)
        Normalization on test set given by CRF (conformal residual fitting).
    norm_calib : NDArray with shape (N, K)
        Normalization on calibration set given by CRF (conformal residual fitting).
    beta_crf : float
        Offset used in NCP e.g. see [Papadoupolous & Haramlambous, 2011].
    sig_lvls: NDArray
        Array with significance levels to construct PI (default=0.05).
    nonconformity_score: str
        Choice of "standard", "deleted" (i.e. jackknife) or "studentized".
    with_bonferroni: bool
        Whether to adjust significance level by n_outputs.

    Returns
    -------
    conf_regions: NDArray with shape (len(sig_lvls), M, n_outputs, 2)
        The confidence region for each test data point and significance level.
    """
    n_outputs = ys_calib.shape[-1]
    if sig_lvls is None:
        sig_lvls = [0.05]
    sig_lvls = np.array(sig_lvls)
    if with_bonferroni:
        sig_lvls = sig_lvls / n_outputs
    conf_regions = []
    for kk in range(n_outputs):
        cp_interval = eval_split_cp(
            ys_calib=ys_calib[:, kk],
            preds_calib=preds_calib[:, kk],
            preds_eval=preds_eval[:, kk],
            h_eval=(h_eval[:, kk, kk] if h_eval is not None else None),
            h_calib=(h_calib[:, kk, kk] if h_calib is not None else None),
            norm_eval=(norm_eval[:, kk] if norm_eval is not None else None),
            norm_calib=(norm_calib[:, kk] if norm_calib is not None else None),
            beta_crf=beta_crf,
            sigma_sq=sigma_sq,
            sig_lvls=sig_lvls,
            nonconformity_score=nonconformity_score,
        )
        conf_regions.append(cp_interval)
    conf_regions = np.array(conf_regions)
    conf_regions = np.moveaxis(conf_regions, 0, -2)
    return conf_regions


def eval_crr_multioutput(
    ys: NDArray,
    preds: NDArray,
    preds_eval: NDArray,
    h_m: NDArray,
    h_mn: NDArray,
    h_n: NDArray,
    sigma_sq: float = 1.0,
    sig_lvls: Optional[List[float]] = None,
    nonconformity_score: str = "standard",
    output_independence: bool = False,
    with_bonferroni: bool = True,
) -> NDArray:
    """Run CRR for vector regression.

    CRR algorithm fixed to use burnaev implementation (for no
    particular reason other than efficiency).

    Parameters (N=n_train, M=n_test, K=n_outputs)
    ----------
    ys: NDArray with shape (N, K)
        Train targets array.
    preds: NDArray with shape (N, K)
        Predictions on train set.
    preds_eval: NDArray with shape (M, K)
        Predictions on test/grid points.
    h_m: NDArray with shape (M, K, K)
        Leverage/marginal variance on test/grid points.
    h_mn: NDArray with shape (M, N, K, K)
        Cross-leverages/variances between test and train points.
    h_n: NDArray with shape (N, K, K)
        Leverage/marginal variance on train set.
    sigma_sq: float
        Observation noise (variance).
    sig_lvls: list[float]
        List of significance levels to construct PI (default=0.05).
    nonconformity_score: str
        Choice of "standard", "deleted" (i.e. jackknife) or "studentized".
    output_independence: bool
        Whether to evaluate cofficients {a, B} using multi-output residual or not
        (full output independence ass.).
    with_bonferroni: bool
        Whether to adjust significance level by n_outputs.

    Returns
    -------
    conf_regions: NDArray with shape (len(sig_lvls), M, n_outputs, 2)
        The confidence region for each test data point and significance level.
    """
    n_outputs = ys.shape[-1]
    if sig_lvls is None:
        sig_lvls = [0.05]
    sig_lvls = np.array(sig_lvls)
    if with_bonferroni:
        sig_lvls = sig_lvls / n_outputs
    if output_independence:
        conf_regions = []
        for kk in range(n_outputs):
            cp_interval = eval_crr(
                ys=ys[:, kk],
                preds=preds[:, kk],
                preds_eval=preds_eval[:, kk],
                h_m=h_m[:, kk, kk],
                h_mn=h_mn[:, :, kk, kk],
                h_n=h_n[:, kk, kk],
                sigma_sq=sigma_sq,
                sig_lvls=sig_lvls,
                nonconformity_score=nonconformity_score,
                algo="burnaev",
            )
            conf_regions.append(cp_interval)

        conf_regions = np.array(conf_regions)
        conf_regions = np.moveaxis(conf_regions, 0, -2)
    else:
        # Multiply "leverages" through by sigma noise.
        h_m = h_m * (1 / sigma_sq)
        h_mn = h_mn * (1 / sigma_sq)
        h_n = h_n * (1 / sigma_sq)

        h_m_plus = h_m + np.eye(n_outputs)[None, :]
        # NB: may have to replace with inverse via Cholesky if numerical issues arise
        b_m = np.array([np.linalg.inv(h_m_plus_single) for h_m_plus_single in h_m_plus])  # (M,K,K)
        b_n = -np.einsum(
            "mnji,mjk->mnik", h_mn, b_m
        )  # with shape (M,N,K,K) # NB: careful h_mn not symmetric

        a_n = np.expand_dims(ys - preds, 0) - np.einsum("mnij,mj->mni", b_n, preds_eval)  # (M,N,K)
        a_m = -np.einsum("mij,mj->mi", b_m, preds_eval)  # (M,K)
        if nonconformity_score in ["deleted", "studentized"]:
            h_m_bar = np.einsum("mij,mjk->mik", h_m, b_m)  # (M,K,K)
            h_n_bar = h_n[None, :] - np.einsum(
                "mnji,mjk,mnkl->mnil", h_mn, b_m, h_mn
            )  # shp (M,N,K,K)
            scale_m = np.eye(n_outputs)[None, :] - h_m_bar
            scale_n = np.eye(n_outputs)[None, None, :] - h_n_bar
            scale_n_inv = np.array([[np.linalg.inv(cov) for cov in mtx] for mtx in scale_n])
            scale_m_inv = np.array([np.linalg.inv(cov) for cov in scale_m])
            if nonconformity_score == "studentized":
                # Square root for studentized variety.
                # If one does Cholesky factorization rather than np.linalg.inv, then one can
                # directly obtain inverse square root and reuses computation.
                scale_n_inv = np.array(
                    [[sp.linalg.sqrtm(mtx_scale) for mtx_scale in mtx] for mtx in scale_n_inv]
                )
                scale_m_inv = np.array([sp.linalg.sqrtm(mtx_scale) for mtx_scale in scale_m_inv])
            b_n = np.einsum("mnij,mnjk->mnik", scale_n_inv, b_n)
            b_m = np.einsum("mij,mjk->mik", scale_m_inv, b_m)
            a_n = np.einsum("mnij,mnj->mni", scale_n_inv, a_n)
            a_m = np.einsum("mij,mj->mi", scale_m_inv, a_m)
        # Take diagonal of B coefficient matrix then use existing (single-output) CRR.
        b_m_diag = b_m[:, np.arange(n_outputs), np.arange(n_outputs)]
        b_n_diag = b_n[:, :, np.arange(n_outputs), np.arange(n_outputs)]
        conf_regions = np.array(
            [
                eval_crr_burnaev(
                    a_n[:, :, kk], a_m[:, kk], b_n_diag[:, :, kk], b_m_diag[:, kk], sig_lvls
                )
                for kk in range(n_outputs)
            ]
        )
        conf_regions = np.moveaxis(conf_regions, 0, -2)
    return conf_regions


def eval_cqr_multioutput(
    ys_calib: NDArray,
    preds_calib: NDArray,
    preds_eval: NDArray,
    sig_lvls: List[float],
    with_bonferroni: bool = True,
) -> NDArray:
    """Run multi-output conformal quantile regression.

    Parameters (N=n_calib, M=n_test, K=n_outputs).
    ----------
    ys_calib: NDArray with shape (N, K)
        Calibation targets array.
    preds_calib: NDArray with shape (N, len(sig_lvls), K, 2)
        Predictions on calibration set.
    preds_eval: NDArray with shape (M, len(sig_lvls), K, 2)
        Predictions on test/grid points.
    sig_lvls: list[float]
        List of significance levels to construct PI (default=0.05).
    with_bonferroni: bool
        Whether to adjust significance level by n_outputs.

    Returns
    -------
    conf_regions: NDArray with shape (len(sig_lvls), M, K, 2)
        The confidence region for each test data point and significance level.
    """
    n_outputs = ys_calib.shape[-1]
    if sig_lvls is None:
        sig_lvls = [0.05]
    sig_lvls = np.array(sig_lvls)
    if with_bonferroni:
        sig_lvls = sig_lvls / n_outputs

    conf_regions = []
    for kk in range(n_outputs):
        cp_interval = eval_cqr(
            ys_calib=ys_calib[:, kk],
            preds_calib=preds_calib[:, kk],
            preds_eval=preds_eval[:, kk],
            sig_lvls=sig_lvls,
        )
        conf_regions.append(cp_interval)

    conf_regions = np.array(conf_regions)
    conf_regions = np.moveaxis(conf_regions, 0, -2)

    return conf_regions


def eval_conf_hyperrectangle_metrics(
    conf_regions: NDArray, ys_test: NDArray, scale: Optional[NDArray] = None
) -> NDArray:
    """Compute average volume and coverage for multidimensional CP intervals.

    Parameters
    ----------
    conf_regions: NDArray with shape (len(sig_lvls), n_test, n_outputs, 2).
        Computed confidence regions.
    ys_test: NDArray with shape (n_test, n_outputs)
        Test labels.
    scale: NDArray with shape (n_outputs,)
        Scale used to preprocess regression data. See RegressionDatasets in datasets_uci.py

    Returns
    -------
    volume: NDArray with shape (len(sig_lvls),)
    coverage: NDArray with shape (len(sig_lvls),)
    """
    n_outputs = ys_test.shape[-1]
    if scale is None:
        scale = np.ones(n_outputs)
    if isinstance(scale, (int, float)):
        scale = scale * np.ones(n_outputs)
    ys_test = ys_test[None, :, :]
    coverage = np.mean(
        np.all(
            (ys_test >= conf_regions[:, :, :, 0]) & (ys_test <= conf_regions[:, :, :, 1]), axis=-1
        ),
        axis=-1,
    )
    volume = np.mean(
        np.prod(
            scale[None, None, :] * (conf_regions[:, :, :, 1] - conf_regions[:, :, :, 0]), axis=-1
        ),
        axis=-1,
    )
    return volume, coverage
