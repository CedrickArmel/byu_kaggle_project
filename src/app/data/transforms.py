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

import monai.transforms as mt
from omegaconf import DictConfig


def get_transforms(cfg: "DictConfig", transform: "str" = "static") -> "mt.Transform":
    if transform not in ["static", "test", "train", "validation"]:
        raise ValueError(
            "transform argument must be one of eval, static, test or train!"
        )
    roi_size = cfg.dataset_args.transforms.roi_size
    batch_size = cfg.dataset_args.transforms.batch_size
    ratios = cfg.dataset_args.transforms.ratios

    if transform.lower() == "static":
        compose = mt.Compose(
            [
                mt.EnsureChannelFirstd(
                    keys=["input", "target"], channel_dim="no_channel"
                ),
                mt.NormalizeIntensityd(keys=["input"]),
                mt.Orientationd(keys=["input", "target"], axcodes="RAS"),
            ]
        )
    elif transform.lower() == "train":
        compose = mt.Compose(
            [
                mt.RandCropByLabelClassesd(
                    keys=["input", "target"],
                    label_key="target",
                    spatial_size=list(roi_size),
                    num_samples=batch_size,
                    num_classes=2,
                    ratios=ratios,
                    warn=False,
                )
            ]
        )
    elif transform.lower() == "validation":
        compose = mt.Compose(
            [
                mt.GridPatchd(
                    keys=["input", "target"],
                    patch_size=list(roi_size),
                    pad_mode="reflect",
                )
            ]
        )
    elif transform.lower() == "test":
        compose = mt.Compose(
            [
                mt.EnsureChannelFirstd(keys=["input"], channel_dim="no_channel"),
                mt.NormalizeIntensityd(keys=["input"]),
                mt.Orientationd(keys=["input"], axcodes="RAS"),
                mt.GridPatchd(
                    keys=["input"], patch_size=list(roi_size), pad_mode="reflect"
                ),
            ]
        )
    return compose
