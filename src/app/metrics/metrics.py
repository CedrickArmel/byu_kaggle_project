# MIT License
#
# Copyright (c) 2024, Yebouet Cédrick-Armel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Derived from:
https://www.kaggle.com/code/metric/czi-cryoet-84969?scriptVersionId=208227222&cellId=1
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import KDTree


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(
    reference_points: "NDArray", candidate_points: "NDArray", reference_radius: "int"
) -> "tuple[int, int, int]":
    """
    Computes the metrics for comparing reference points and candidate points within a given radius.
    Args:
        reference_points (NDArray): An array of reference points representing the ground truth.
        candidate_points (NDArray): An array of candidate points to be evaluated.
        reference_radius (int): The radius within which a candidate point is considered a match to a reference point.
    Returns:
        tuple[int, int, int]: A tuple containing:
            - tp (int): The number of true positives (candidate points that match reference points within the radius).
            - fp (int): The number of false positives (candidate points that do not match any reference points).
            - fn (int): The number of false negatives (reference points that do not have any matching candidate points).
    """
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = list(set(matches_within_threshold))
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
    cfg: "SimpleNamespace", sol: "pd.DataFrame", sub: "pd.DataFrame"
) -> "tuple[int, ...]":
    """
    F-beta
        - a true positive occurs when the predicted location is within a threshold of the motor_radius
        - raw results (TP, FP, FN) are aggregated across all tomograms
    """

    motor_radius = cfg.motor_radius * cfg.dt_multiplier

    # Filter submission to only contain experiments found in the solution split
    split_ids = set(sol["tomo_id"].unique())
    submission = sub.loc[sub["tomo_id"].isin(split_ids)]
    solution = sol

    assert solution.duplicated(subset=["tomo_id", "z", "y", "x"]).sum() == 0

    results = {
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
    }

    for tid in split_ids:
        select = solution["tomo_id"] == tid
        vxs = solution[select]["vxs"].iloc[0]
        reference_points = solution.loc[select, ["z", "y", "x"]].values

        select = submission["tomo_id"] == tid
        candidate_points = submission.loc[select, ["z", "y", "x"]].values
        reference_radius = int(
            (motor_radius / vxs) * 2
        )  # TODO: dois-je prendre en compte le vxs ?

        if len(reference_points) == 0:
            reference_points = np.array([])
            reference_radius = 1
        if len(candidate_points) == 0:
            candidate_points = np.array([])

        tp, fp, fn = compute_metrics(
            reference_points, candidate_points, reference_radius
        )
        results["total_tp"] += tp
        results["total_fp"] += fp
        results["total_fn"] += fn

    tp = results["total_tp"]
    fp = results["total_fp"]
    fn = results["total_fn"]
    return tp, fp, fn


def thresholder(
    c: "float",
    cfg: "SimpleNamespace",
    solution: "pd.DataFrame",
    submission: "pd.DataFrame",
) -> "float":
    beta = cfg.score_beta
    pos_submission = submission[submission["conf"] > c].copy()
    pos_solution = solution[solution["z"] >= 0].copy()
    pos_sub_tomo = set(pos_submission["tomo_id"].unique())
    pos_sol_tomo = set(pos_solution["tomo_id"].unique())
    pos_fp = len(pos_sub_tomo - pos_sol_tomo)  # neg_fn = pos_fp
    pos_fn = len(pos_sol_tomo - pos_sub_tomo)  # neg_fp = pos_fn

    neg_sub_tomo = set(submission[submission["conf"] <= c]["tomo_id"].unique())
    neg_sub_tomo = neg_sub_tomo - pos_sub_tomo
    neg_sol_tomo = set(solution[solution["z"] < 0]["tomo_id"].unique())
    neg_tp = len(neg_sub_tomo & neg_sol_tomo)

    candidates = pos_submission.loc[
        pos_submission["tomo_id"].isin(pos_sub_tomo & pos_sol_tomo)
    ].copy()
    sc_tp, sc_fp, sc_fn = score(cfg, sol=pos_solution, sub=candidates)
    tp = neg_tp + sc_tp
    fp = pos_fp + sc_fp
    fn = pos_fn + sc_fn
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    fbeta: "float" = (
        ((1 + beta**2) * (prec * rec) / (beta**2 * prec + rec))
        if (prec + rec) > 0
        else 0.0
    )
    return fbeta


def get_final_submission(
    submission: "pd.DataFrame", best_th: "float"
) -> "pd.DataFrame":
    """Remove false positives and negatives from the submission dataframe"""
    select = submission["conf"] > best_th
    pos_tomo = set(submission[select]["tomo_id"].unique())
    select = submission["conf"] < best_th
    neg_tomo = set(submission[select]["tomo_id"].unique())
    neg_tomo = neg_tomo - pos_tomo
    submission.loc[submission["tomo_id"].isin(neg_tomo), ["z", "y", "x"]] = -1
    submission = submission.drop(columns=["conf"])
    return submission


def calc_metric(
    cfg: "SimpleNamespace", pred_df: "pd.DataFrame", val_df: "pd.DataFrame"
) -> "tuple[float, float]":
    """
    Calculate the best threshold and score for the given prediction and validation dataframes.
    Args:
        cfg (types.SimpleNamespace): Configuration object containing project's parameters.
        pred_df (pd.DataFrame): Dataframe containing the predictions.
        val_df (pd.DataFrame): Dataframe containing the validation data.
    Returns:
        tuple: A tuple containing the best score and the best threshold.
    """
    solution = val_df.copy()
    submission = pred_df.copy()
    scores = []
    ths = np.arange(0, cfg.max_th, 0.005)
    for c in ths:
        scores += [thresholder(c, cfg, solution, submission)]
    best_idx = int(np.argmax(scores))
    best_th = float(ths[best_idx])
    best_score = float(scores[best_idx])
    return best_score, best_th
