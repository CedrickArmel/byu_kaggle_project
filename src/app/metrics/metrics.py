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

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from omegaconf import DictConfig
from scipy.spatial import KDTree
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class BYUFbeta(Metric):
    def __init__(self, cfg: "DictConfig", **kwargs) -> "None":
        super().__init__(**kwargs)
        self.add_state(name="preds", default=[], dist_reduce_fx="cat")
        self.add_state(name="targets", default=[], dist_reduce_fx="cat")
        self.cfg = cfg

    def update(self, pred: "torch.Tensor", target: "torch.Tensor") -> "None":
        self.preds.append(pred)  # type: ignore[operator, union-attr]
        self.targets.append(target)  # type: ignore[operator, union-attr]

    def compute(self) -> "dict[str, float]":
        preds: "torch.Tensor" = dim_zero_cat(x=self.preds)  # type: ignore[arg-type]
        targets: "torch.Tensor" = dim_zero_cat(x=self.targets)  # type: ignore[arg-type]
        targets = torch.unique(targets, dim=0)
        preds = get_topk_by_id(preds=preds, targets=targets)

        fbeta1s: "list" = []
        fbeta2s: "list" = []
        ths: "NDArray" = np.arange(start=0, stop=self.cfg.max_th, step=0.001)

        for t in ths:
            fbeta1, fbeta2 = self.score_fn(t=t, preds=preds, targets=targets)
            fbeta1s += [fbeta1]
            fbeta2s += [fbeta2]

        best_idx = int(np.argmax(a=fbeta1s))
        fb1_thd = float(ths[best_idx])
        best_fbeta1 = float(fbeta1s[best_idx])

        best_idx = int(np.argmax(a=fbeta2s))
        fb2_thd = float(ths[best_idx])
        best_fbeta2 = float(fbeta2s[best_idx])

        return dict(
            fbeta1=best_fbeta1, thd1=fb1_thd, fbeta2=best_fbeta2, thd2=fb2_thd)


    def score_fn(
        self, t: "float", preds: "torch.Tensor", targets: "torch.Tensor"
    ) -> "tuple[float,...]":
        """Computes the scores"""
        beta: "float" = self.cfg.score_beta
        ut_preds, candidates, ntargets, ptargets = thresholder(t, preds, targets)
        tp2, fp2, fn2 = filter_negatives(ut_preds, ntargets)
        tp1, fp1, fn1 = self.compute_candidates_cm_metrics(candidates, ptargets)

        prec1: "float" = tp1 / (tp1 + fp1) if tp1 + fp1 > 0 else 0.0
        rec1: "float" = tp1 / (tp1 + fn1) if tp1 + fn1 > 0 else 0.0

        prec2: "float" = tp2 / (tp2 + fp2) if tp2 + fp2 > 0 else 0.0
        rec2: "float" = tp2 / (tp2 + fn2) if tp2 + fn2 > 0 else 0.0

        fbeta1: "float" = (
            ((1 + beta**2) * (prec1 * rec1) / (beta**2 * prec1 + rec1))
            if (prec1 + rec1) > 0
            else 0.0
        )

        fbeta2: "float" = (
            ((1 + beta**2) * (prec2 * rec2) / (beta**2 * prec2 + rec2))
            if (prec2 + rec2) > 0
            else 0.0
        )
        return fbeta1, fbeta2

    def compute_candidates_cm_metrics(
        self, candidates: "torch.Tensor", ptargets: "torch.Tensor"
    ) -> "tuple[int, ...]":
        motor_radius: "float" = self.cfg.motor_radius * self.cfg.dt_multiplier
        tp1, fp1, fn1 = 0, 0, 0

        tomo_ids: "list[float]" = ptargets[:, -2].unique().tolist()

        for tid in tomo_ids:
            # Sélectionne les points du tomogram courant
            ref_select: "torch.Tensor" = ptargets[:, -2] == tid
            candidate_select: "torch.Tensor" = candidates[:, -1] == tid

            reference_points: "torch.Tensor" = ptargets[ref_select, :-2]
            candidate_points: "torch.Tensor" = candidates[candidate_select, :-1]
            vxs: "torch.Tensor" = ptargets[ref_select, -2][0]  # scalar tensor
            reference_radius = int((motor_radius / vxs) * 2)

            if len(candidate_points) == 0:
                fn1 += len(reference_points)
                continue

            reference_points_np: "NDArray" = reference_points.cpu().numpy()
            candidate_points_np: "NDArray" = candidate_points.cpu().numpy()

            ref_tree = KDTree(reference_points_np)
            cand_tree = KDTree(candidate_points_np)
            raw_matches: "ArrayLike" = cand_tree.query_ball_tree(
                ref_tree, r=reference_radius
            )

            matched_references = []
            for match in raw_matches:
                matched_references.extend(match)

            matched_references = list(set(matched_references))
            tp1 += len(matched_references)
            fp1 += len(candidate_points) - len(matched_references)
            fn1 += len(reference_points) - len(matched_references)
        return tp1, fp1, fn1


def filter_negatives(
    ut_preds: "torch.Tensor", ntargets: "torch.Tensor"
) -> "tuple[int, ...]":
    tp2, fp2, fn2 = 0, 0, 0
    reference_ids: "list[float]" = ntargets[:, -2].unique().tolist()
    candidates_ids: "list[float]" = ut_preds[:, -1].unique().tolist()
    tp2 += len(set(reference_ids) & set(candidates_ids))
    fn2 += len(reference_ids) - tp2  # over threshold from empty
    missing_ids: "set[float]" = set(candidates_ids) - set(reference_ids)
    for i in missing_ids:
        fp2 += ut_preds[ut_preds[:, -1] == i].size(0)
    return tp2, fp2, fn2


def get_topk_by_id(preds: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    """Returns the k most confident points by tomo_id."""
    topk_results: "list[torch.Tensor]" = []
    ids = targets[:, -2].unique()
    for i in ids:
        k = targets[targets[:, -2] == i].size(dim=0)
        subset: "torch.Tensor" = preds[preds[:, -2] == i]
        if subset.size(dim=0) <= k:
            topk: "torch.Tensor" = subset
        else:
            _, select = torch.topk(input=subset[:, -2], k=k)
            topk = subset[select]
        topk_results.append(topk)
    return torch.cat(topk_results, dim=0)


def thresholder(
    t: "float", preds: "torch.Tensor", targets: "torch.Tensor"
) -> "tuple[torch.Tensor, ...]":
    ut_preds: "torch.Tensor" = torch.unique(
        preds[torch.where(preds[:, -1] < t)][:, :-1], dim=0
    )
    ot_preds: "torch.Tensor" = torch.unique(
        preds[torch.where(preds[:, -1] >= t)][:, :-1], dim=0
    )
    ntargets: "torch.Tensor" = torch.unique(
        targets[torch.where(targets[:, 0] == -1)], dim=0
    )
    ptargets: "torch.Tensor" = torch.unique(
        targets[torch.where(targets[:, 0] >= 0)], dim=0
    )

    select_candidates: "torch.Tensor" = torch.isin(ot_preds[:, -1], ptargets[:, -2])
    candidates: "torch.Tensor" = ot_preds[select_candidates]
    return ut_preds, candidates, ntargets, ptargets
