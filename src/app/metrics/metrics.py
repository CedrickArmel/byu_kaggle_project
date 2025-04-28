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
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class ParticipantVisibleError(Exception):
    pass


class BYUFbeta(Metric):
    def __init__(
        self, cfg: "SimpleNamespace", tn_as_tp: "bool" = False, **kwargs
    ) -> "None":
        super().__init__(**kwargs)
        self.add_state(name="preds", default=[], dist_reduce_fx="cat")
        self.add_state(name="targets", default=[], dist_reduce_fx="cat")
        self.cfg = cfg
        self.tn_as_tp = tn_as_tp

    def update(self, pred: "torch.Tensor", target: "torch.Tensor") -> "None":
        self.preds.append(pred)  # type: ignore[operator, union-attr]
        self.targets.append(target)  # type: ignore[operator, union-attr]

    def compute(self) -> "dict[str, float]":
        preds = dim_zero_cat(self.preds)  # type: ignore[arg-type]
        targets = dim_zero_cat(self.targets)  # type: ignore[arg-type]
        scores = []
        ths = np.arange(0, self.cfg.max_th, 0.005)
        for t in ths:
            scores += [self._score(t, preds, targets)]
            best_idx = int(np.argmax(scores))
        best_th = float(ths[best_idx])
        best_score = float(scores[best_idx])
        return dict(byu_score=best_score, best_ths=best_th)

    def _score(
        self, t: "float", preds: "torch.Tensor", targets: "torch.Tensor"
    ) -> "float":
        beta = self.cfg.score_beta
        ut_preds, ot_preds, ntargets, ptargets = self._thresholder(t, preds, targets)
        candidates, fp_ = self._compute_candidates(ot_preds, ptargets)
        tn, fn_ = self._filter_negatives(ut_preds, ntargets)
        tp, fp, fn = self._compute_candidates_cm_metrics(candidates, ptargets)
        if self.tn_as_tp:
            tp += tn
        fp += fp_
        fn += fn_
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta: "float" = (
            ((1 + beta**2) * (prec * rec) / (beta**2 * prec + rec))
            if (prec + rec) > 0
            else 0.0
        )
        return fbeta

    def _thresholder(
        self, t: "float", preds: "torch.Tensor", targets: "torch.Tensor"
    ) -> "tuple[torch.Tensor, ...]":
        ut_preds = torch.unique(preds[torch.where(preds[:, -1] < t)][:, :-1], dim=0)
        ot_preds = torch.unique(preds[torch.where(preds[:, -1] >= t)][:, :-1], dim=0)
        targets = torch.unique(targets)
        ntargets = torch.unique(targets[torch.where(targets[:, 0] == -1)], dim=0)
        ptargets = torch.unique(targets[torch.where(targets[:, 0] >= 0)], dim=0)
        return ut_preds, ot_preds, ntargets, ptargets

    def _compute_candidates(
        self, ot_preds: "torch.Tensor", ptargets: "torch.Tensor"
    ) -> "tuple[torch.Tensor, int]":
        select_candidates = torch.isin(ot_preds[:, -1], ptargets[:, -2])
        candidates = ot_preds[select_candidates]
        fp = len(ot_preds[~select_candidates])
        return candidates, fp

    def _filter_negatives(
        self, ut_preds: "torch.Tensor", ntargets: "torch.Tensor"
    ) -> "tuple[int, int]":
        select_negatives = torch.isin(
            ut_preds[:, -1], ntargets[:, -2]
        )  # z, y, x , ids, vxs
        tn = len(ut_preds[select_negatives])
        fn = len(ut_preds[~select_negatives])
        return tn, fn

    def _compute_candidates_cm_metrics(
        self, candidates: "torch.Tensor", ptargets: "torch.Tensor"
    ) -> "tuple[int, ...]":
        motor_radius: "float" = self.cfg.motor_radius * self.cfg.dt_multiplier
        tp = 0
        fp = 0
        fn = 0
        tomo_ids = ptargets[:, -2].unique()

        for tid in tomo_ids:
            # Sélectionne les points du tomogram courant
            ref_select: "torch.Tensor" = ptargets[:, -2] == tid
            candidate_select = candidates[:, -1] == tid

            reference_points = ptargets[
                ref_select, :-2
            ]  # On enlève l'id (seulement coords : z,y,x)
            candidate_points = candidates[candidate_select, :-1]
            vxs = ptargets[ref_select, -2][0]
            reference_radius = int((motor_radius / vxs) * 2)

            if len(reference_points) == 0:
                tp += 0
                fp += len(candidate_points)
                fn += 0
                continue

            if len(candidate_points) == 0:
                tp += 0
                fp += 0
                fn += len(reference_points)
                continue

            reference_points_np: "NDArray" = reference_points.cpu().numpy()
            candidate_points_np: "NDArray" = candidate_points.cpu().numpy()

            ref_tree = KDTree(reference_points_np)
            cand_tree = KDTree(candidate_points_np)
            raw_matches = cand_tree.query_ball_tree(ref_tree, r=reference_radius)

            matched_references = []
            for match in raw_matches:
                matched_references.extend(match)

            matched_references = list(set(matched_references))
            tp += len(matched_references)
            fp += len(candidate_points) - len(matched_references)
            fn += len(reference_points) - len(matched_references)
        return tp, fp, fn


# TODO: torchmetrics.utilities.distributed.gather_all_tensors
