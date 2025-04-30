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

    if transform.lower() == "static":
        compose = mt.Compose(
            [
                mt.EnsureChannelFirstd(
                    keys=["input", "target"], channel_dim="no_channel"
                ),
                mt.AdjustContrastd(keys=["input"], gamma=cfg.gamma),
                mt.ScaleIntensityd(keys=["input"]),
                mt.Orientationd(keys=["input", "target"], axcodes="RAS"),
            ]
        )
    elif transform.lower() == "train":
        compose = mt.Compose(
            [
                mt.RandCropByLabelClassesd(
                    keys=["input", "target"],
                    label_key="target",
                    spatial_size=cfg.roi_size,
                    num_samples=cfg.sub_batch_size,
                    num_classes=2,
                    ratios=[1, 1],
                    warn=False,
                ),
                mt.RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=2),
                mt.RandRotate90d(
                    keys=["input", "target"], prob=0.75, max_k=3, spatial_axes=(0, 1)
                ),
                mt.RandRotated(
                    keys=["input", "target"],
                    prob=0.5,
                    range_x=0.78,
                    range_y=0.0,
                    range_z=0.0,
                    padding_mode="reflection",
                ),
            ]
        )
    elif transform.lower() == "validation":
        compose = mt.Compose(
            [
                mt.GridPatchd(
                    keys=["input", "target"],
                    patch_size=cfg.roi_size,
                    pad_mode="reflect",
                )
            ]
        )
    elif transform.lower() == "test":
        compose = mt.Compose(
            [
                mt.EnsureChannelFirstd(keys=["input"], channel_dim="no_channel"),
                mt.AdjustContrastd(keys=["input"], gamma=cfg.gamma),
                mt.ScaleIntensityd(keys=["input"]),
                mt.Orientationd(keys=["input"], axcodes="RAS"),
                mt.GridPatchd(
                    keys=["input"], patch_size=cfg.roi_size, pad_mode="reflect"
                ),
            ]
        )
    return compose
