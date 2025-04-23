# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/EugeneNdiaye/homotopy_conformal_prediction/tree/master
# License is provided for attribution purposes only, Not a Contribution.

# Copyright (c) 2019 Happy new year :-)
# anonymous author.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the developers nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Function implementing Nouretdinov's Conformal Ridge Regression (CRR) variant."""

from typing import List

import numpy as np
import portion as intervals
from numpy.typing import NDArray

JITTER = 1e-4


def eval_crr_nouretdinov(
    a_n: NDArray, a_m: float, b_n: NDArray, b_m: float, sig_lvls: List[float]
) -> NDArray:
    """Run Conformalized Ridge Regression (CRR) of Nouretdinov et al.

    Run Conformalized Ridge Regression (CRR) algorithm as proposed in [Nouretdinov et al., 2001].
    Return prediction interval given by convex closure of union over changepoint intervals.

    Residual for "N+1" AOI (+LOO) ridge problem ` A + B*y` where A = [a_n, a_m] and B = [b_n, b_m].

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
    # Enforce B >= 0
    mask = np.where(b_n < 0)[0]
    b_n[mask] *= -1
    a_n[mask] *= -1
    if b_m < 0:
        b_m *= -1
        a_m *= -1
    sets_per_datapoint, changepts_lower, changepts_upper = [], [], []  # S,U,V
    # Excl. (N+1)th point.
    for ii in range(n_data):
        # Constructing set of changepoints.
        if b_n[ii] != b_m:
            changept_1 = (a_n[ii] - a_m) / np.clip(b_m - b_n[ii], JITTER, None)
            changept_2 = -(a_n[ii] + a_m) / np.clip(b_m + b_n[ii], JITTER, None)
            u_i, v_i = (
                (changept_1, changept_2) if changept_1 < changept_2 else (changept_2, changept_1)
            )
            changepts_lower += [u_i]
            changepts_upper += [v_i]
        elif (b_n[ii] != 0) and (a_n[ii] != a_m):
            uv_i = -0.5 * (a_n[ii] + a_m) / np.clip(b_n[ii], JITTER, None)
            changepts_lower += [uv_i]
            changepts_upper += [uv_i]
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
    changepts_sorted = np.sort([-np.inf] + changepts_lower + changepts_upper + [np.inf])
    n_changepts = len(changepts_sorted)
    # Original version from [Nouretdinov et. al., 2001] by taking union of closed intervals.
    # A bit strange forming closed interval at {-inf,+inf} endpoints.
    cp_intervals = []
    for sig_lvl in sig_lvls:
        cp_interval = intervals.empty()
        for jj in range(n_changepts - 1):
            # Compute rank of each changepoint interval.
            n_pvalue_j = 0
            intvl_j = intervals.closed(changepts_sorted[jj], changepts_sorted[jj + 1])
            for _, set_dp in enumerate(sets_per_datapoint):
                n_pvalue_j += intvl_j in set_dp
            # Trivially convert rank to p-value of changepoint interval.
            pvalue_j = n_pvalue_j / (n_data + 1)
            if pvalue_j > sig_lvl:
                cp_interval = cp_interval.union(intvl_j)
        cp_intervals.append(
            [cp_interval.lower.item(), cp_interval.upper.item()]
        )  # Only return "convex closure".
    return np.array(cp_intervals)
