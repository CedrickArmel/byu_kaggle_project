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

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets.flexible_unet import (
    FLEXUNET_BACKBONE,
    SegmentationHead,
    UNetDecoder,
)
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Beta


class PatchedUNetDecoder(UNetDecoder):  # type: ignore[misc]
    """add functionality to output all feature maps"""

    def forward(
        self, features: "list[torch.Tensor]", skip_connect: "int" = 4
    ) -> "list[torch.Tensor]":
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        out = []
        x = features[0]
        out += [x]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)
            out += [x]
        return out


class FlexibleUNet(nn.Module):  # type: ignore[misc]
    """
    A flexible implementation of UNet-like encoder-decoder architecture.
    (Adjusted to support PatchDecoder and multi segmentation heads)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: str | tuple[str, dict[str, Any]] = (
            "batch",
            {"eps": 1e-3, "momentum": 0.1},
        ),
        act: str | tuple[str, dict[str, Any]] = ("relu", {"inplace": True}),
        dropout: float | tuple[str, dict[str, Any]] = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
    ) -> None:
        """
        A flexible implement of UNet, in which the backbone/encoder can be replaced with
        any efficient or residual network. Currently the input must have a 2 or 3 spatial dimension
        and the spatial size of each dimension must be a multiple of 32 if is_pad parameter
        is False.
        Please notice each output of backbone must be 2x downsample in spatial dimension
        of last output. For example, if given a 512x256 2D image and a backbone with 4 outputs.
        Spatial size of each encoder output should be 256x128, 128x64, 64x32 and 32x16.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            backbone: name of backbones to initialize, only support efficientnet and resnet right now,
                can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2, resnet10, ..., resnet200].
            pretrained: whether to initialize pretrained weights. ImageNet weights are available for efficient networks
                if spatial_dims=2 and batch norm is used. MedicalNet weights are available for residual networks
                if spatial_dims=3 and in_channels=1. Default to False.
            decoder_channels: number of output channels for all feature maps in decoder.
                `len(decoder_channels)` should equal to `len(encoder_channels) - 1`,default
                to (256, 128, 64, 32, 16).
            spatial_dims: number of spatial dimensions, default to 2.
            norm: normalization type and arguments, default to ("batch", {"eps": 1e-3,
                "momentum": 0.1}).
            act: activation type and arguments, default to ("relu", {"inplace": True}).
            dropout: dropout ratio, default to 0.0.
            decoder_bias: whether to have a bias term in decoder's convolution blocks.
            upsample: upsampling mode, available options are``"deconv"``, ``"pixelshuffle"``,
                ``"nontrainable"``.
            pre_conv:a conv block applied before upsampling. Only used in the "nontrainable" or
                "pixelshuffle" mode, default to `default`.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            is_pad: whether to pad upsampling features to fit features from encoder. Default to True.
                If this parameter is set to "True", the spatial dim of network input can be arbitrary
                size, which is not supported by TensorRT. Otherwise, it must be a multiple of 32.
        """
        super().__init__()

        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid model_name {backbone} found, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        encoder = FLEXUNET_BACKBONE.register_dict[backbone]
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_parameters = encoder["parameter"]

        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError(
                "The backbone init method must have spatial_dims, in_channels and pretrained parameters."
            )

        encoder_feature_num = encoder["feature_number"]

        if encoder_feature_num > 5:
            raise ValueError(
                "Flexible unet can only accept no more than 5 encoder feature maps."
            )

        decoder_channels = decoder_channels[:encoder_feature_num]

        self.skip_connect = encoder_feature_num - 1

        encoder_parameters.update(
            {
                "spatial_dims": spatial_dims,
                "in_channels": in_channels,
                "pretrained": pretrained,
            }
        )
        encoder_channels = tuple([in_channels] + list(encoder["feature_channel"]))
        encoder_type = encoder["type"]
        self.encoder = encoder_type(**encoder_parameters)

        self.decoder = PatchedUNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=pre_conv,
            align_corners=None,
            is_pad=is_pad,
        )
        # Instanciate a segmentation head for each feature maps outputed by PatchedUNetDecoder
        self.segmentation_heads = nn.ModuleList(
            [
                SegmentationHead(
                    spatial_dims=spatial_dims,
                    in_channels=decoder_channel,
                    out_channels=out_channels + 1,
                    kernel_size=3,
                    act=None,
                )
                for decoder_channel in decoder_channels[:-1]
            ]
        )

    def forward(self, inputs: torch.Tensor) -> "list[torch.Tensor]":
        """
        Performs a forward pass through the model.
        Args:
            inputs (torch.Tensor): The input tensor to the model.
        Returns:
            list[torch.Tensor]: A list of tensors representing the segmented
            feature maps produced by the segmentation heads.
        """
        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)[1:-1]
        x_seg = [
            self.segmentation_heads[i](decoder_out[i]) for i in range(len(decoder_out))
        ]  # segment each feature map
        return x_seg


def count_parameters(model: "nn.Module") -> "int":
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_format(num: "float") -> str:
    """Convert a number to a human-readable format with SI prefixes."""
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


class Mixup(nn.Module):  # type: ignore[misc]
    """Mixup augmentation for 3D data."""

    def __init__(self, mix_beta: "float", mixadd: "bool" = False) -> None:
        """Initialize the Mixup module."""
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(
        self, X: "torch.Tensor", Y: "torch.Tensor", Z: "torch.Tensor | None" = None
    ) -> "tuple[torch.Tensor, ...]":
        """Apply mixup augmentation to the input data."""
        bs = X.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,) * (X.ndim - 1))
        Y_coeffs = coeffs.view((-1,) + (1,) * (Y.ndim - 1))
        X = X_coeffs * X + (1 - X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]
        if Z:
            return X, Y, Z
        return X, Y


class DenseCrossEntropy(nn.Module):  # type: ignore[misc]
    def __init__(self, class_weights: "torch.Tensor | None" = None) -> None:
        """Initialize the DenseCrossEntropy loss function."""
        super(DenseCrossEntropy, self).__init__()
        self.class_weights = class_weights

    def forward(
        self, x: "torch.Tensor", target: "torch.Tensor"
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float)
        loss = -logprobs * target
        class_losses = loss.mean((0, 2, 3, 4))
        if self.class_weights is not None:
            loss = (
                class_losses * self.class_weights.to(class_losses.device)
            ).sum()  # / class_weights.sum()
        else:
            loss = class_losses.sum()
        return loss, class_losses


def to_ce_target(y: "torch.Tensor") -> "torch.Tensor":
    """Convert the target to a format suitable for cross-entropy loss."""
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y, y_bg], 1)
    y = y / y.sum(1, keepdim=True)
    return y


class Net(nn.Module):  # type: ignore[misc]
    """Adapted from ChristofHenkel/kaggle-cryoet-1st-place-segmentation/models
    to support sub_batches and avoid OOM errors.
    """

    def __init__(self, cfg: "DictConfig") -> None:
        """Initialize the Net module."""
        super(Net, self).__init__()
        self.cfg = cfg
        self.backbone = FlexibleUNet(**cfg.backbone_args)
        self.mixup = Mixup(cfg.mixup_beta)
        self.lvl_weights = torch.from_numpy(np.array(cfg.lvl_weights))
        self.loss_fn = DenseCrossEntropy(
            class_weights=torch.from_numpy(np.array(cfg.class_weights))
        )

    def forward(self, batch: "dict[str, Any]") -> "dict[str, Any]":
        """Perform a forward pass through the model."""
        bs = (
            self.cfg.virt_sub_batch_size
            if self.training
            else self.cfg.virt_eval_sub_batch_size
        )

        if bs == -1:
            sub_batches = [batch]
        else:
            sub_batches = [
                {key: value[i : i + bs] for key, value in batch.items()}
                for i in range(0, len(batch["input"]), bs)
            ]

        outputs = {}
        all_logits = []
        all_losses = []

        has_target = "target" in batch  # Check only once

        for b in sub_batches:
            x = b["input"].to(torch.float32)
            y = b["target"].to(torch.float32) if has_target else None

            if self.training and has_target and torch.rand(1)[0] < self.cfg.mixup_p:
                x, y = self.mixup(x, y)

            out = self.backbone(x)

            if not self.training:
                all_logits.append(out[-1])

            if has_target:
                ys = [F.adaptive_max_pool3d(y, o.shape[-3:]) for o in out]
                loss_values = torch.stack(
                    [
                        self.loss_fn(out[i], to_ce_target(ys[i]))[0]
                        for i in range(len(out))
                    ]
                )
                lvl_weights = self.lvl_weights.to(loss_values.device)
                weighted_loss = (loss_values * lvl_weights).sum() / lvl_weights.sum()
                all_losses.append(weighted_loss)

        if has_target:
            outputs["loss"] = torch.stack(all_losses).mean()

        if not self.training:
            outputs["logits"] = torch.cat(all_logits, dim=0)
            if "location" in batch:
                outputs["location"] = batch["location"]
            if "scale" in batch:
                outputs["scale"] = batch["scale"]
            if "id" in batch:
                outputs["id"] = batch["id"]
            if "dims" in batch:
                outputs["dims"] = batch["dims"]
        return outputs
