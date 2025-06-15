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

import random
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.networks.nets.basic_unet import UpCat
from monai.networks.nets.flexible_unet import FLEXUNET_BACKBONE, SegmentationHead
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Beta

from app.losses import FocalLoss


class PatchedUNetDecoder(nn.Module):  # type: ignore[misc]
    """add functionality to output all feature maps"""

    def __init__(
        self,
        spatial_dims: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        act: str | tuple,
        norm: str | tuple,
        dropout: float | tuple,
        bias: bool,
        upsample: str,
        pre_conv: str | None,
        interp_mode: str,
        align_corners: bool | None,
        is_pad: bool,
        skip_connect: int = 4,
    ):
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError(
                "the length of `encoder_channels` should be no less than 2."
            )
        if len(decoder_channels) < 1:
            raise ValueError("`len(decoder_channels)` should be no less than 1 `.")
        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        blocks = []

        for in_chn, skip_chn, out_chn, halve in zip(
            in_channels, skip_channels, decoder_channels, halves
        ):
            blocks.append(
                UpCat(
                    spatial_dims=spatial_dims,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                    is_pad=is_pad,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.skip_connect = min(skip_connect, len(self.blocks))

    def forward(self, features: "list[torch.Tensor]") -> "list[torch.Tensor]":
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        out = []
        x = features[0]
        out += [x]
        for i, block in enumerate(self.blocks):
            if i < self.skip_connect:
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
        act: str | tuple[str, dict[str, Any]] = (
            "leakyrelu",
            {"negative_slope": 0.1, "inplace": True},
        ),
        dropout: float | tuple[str, dict[str, Any]] = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
        skip_connect: int = 4,
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
        encoder_parameters = encoder["parameter"]
        encoder_feature_num = encoder["feature_number"]

        self.backbone = backbone
        self.spatial_dims = spatial_dims

        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError(
                "The backbone init method must have spatial_dims, in_channels and pretrained parameters."
            )

        if encoder_feature_num > 5:
            raise ValueError(
                "Flexible unet can only accept no more than 5 encoder feature maps."
            )

        decoder_channels = decoder_channels[:encoder_feature_num]
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
            skip_connect=skip_connect,
        )

        self.segmentation_heads = nn.ModuleList(
            [
                SegmentationHead(
                    spatial_dims=spatial_dims,
                    in_channels=decoder_channel,
                    out_channels=out_channels,
                    kernel_size=3,
                    act=None,
                )
                for decoder_channel in decoder_channels
            ]
        )

    def forward(self, inputs: "torch.Tensor") -> "list[torch.Tensor]":
        """
        Performs a forward pass through the model.
        Args:
            inputs (torch.Tensor): The input tensor to the model.
        Returns:
            list[torch.Tensor]: A list of tensors representing the segmented
            feature maps produced by the segmentation heads.
        """
        x: "torch.Tensor" = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out)[1:]
        x_seg = [
            self.segmentation_heads[i](decoder_out[i]) for i in range(len(decoder_out))
        ]
        return x_seg


class Mixup(nn.Module):  # type: ignore[misc]
    """Mixup augmentation for data."""

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


class CutmixSimple(nn.Module):
    """Simple cutmix augmentation for data."""

    def __init__(self, cut_beta: "float" = 5.0, cut_dims: "tuple" = (-2, -1)):
        super().__init__()
        assert all(_ < 0 for _ in cut_dims), "dims must be negatively indexed."
        self.beta_distribution = Beta(cut_beta, cut_beta)  # beta = 5 = gaussianlike
        self.dims = cut_dims

    def forward(self, X, Y, Z=None):
        cut_idx = self.beta_distribution.sample().item()

        perm = torch.randperm(X.size(0))
        X_perm = X[perm]
        Y_perm = Y[perm]

        axis = random.choice(self.dims)

        # Get cut idxs
        cutoff_X = int(cut_idx * X.shape[axis])
        cutoff_Y = int(cut_idx * Y.shape[axis])

        # Apply cut
        if axis == -1:
            X[..., :cutoff_X] = X_perm[..., :cutoff_X]
            Y[..., :cutoff_Y] = Y_perm[..., :cutoff_Y]
        elif axis == -2:
            X[..., :cutoff_X, :] = X_perm[..., :cutoff_X, :]
            Y[..., :cutoff_Y, :] = Y_perm[..., :cutoff_Y, :]
        else:
            raise ValueError("CutmixSimple: Axis not implemented.")

        return X, Y


class Net(nn.Module):  # type: ignore[misc]
    """Adapted from ChristofHenkel/kaggle-cryoet-1st-place-segmentation/models
    to support sub_batches and avoid OOM errors.
    """

    def __init__(self, cfg: "DictConfig") -> None:
        """Initialize the Net module."""
        super(Net, self).__init__()
        self.cfg = cfg
        self.backbone = FlexibleUNet(**cfg.backbone_args)
        self.mixup = Mixup(mix_beta=cfg.mixup_beta)
        self.cutmix = CutmixSimple(cut_beta=cfg.cut_beta, cut_dims=cfg.cut_dims)
        self.loss_fn = FocalLoss(**self.cfg.loss_args)
        self.dice_fn = DiceLoss(**self.cfg.dice_args)
        self.max_loss_fn = FocalLoss(**self.cfg.max_loss_args)
        self.avg_loss_fn = FocalLoss(**self.cfg.avg_loss_args)
        self.loss_contributions = torch.tensor(list(self.cfg.loss_contributions))

    def forward(self, batch: "dict[str, Any]") -> "dict[str, Any]":
        """Perform a forward pass through the model."""
        bs = (
            self.cfg.virt_sub_batch_size
            if self.training
            else self.cfg.virt_eval_sub_batch_size
        )
        has_target = "target" in batch
        full_size = batch["input"].shape[0]
        device: "torch.device" = batch["input"].device

        target: "torch.Tensor" = torch.empty(
            0, device=device
        )  # better than empty list because of XLA compilation performance
        all_outs = []
        outputs = {}

        for i in range(0, full_size, bs if bs != -1 else full_size):
            x: "torch.Tensor " = batch["input"][i : i + bs].float()
            y: "torch.Tensor | None" = (
                batch["target"][i : i + bs].float() if has_target else None
            )

            if self.training:  # we assume a target is always present during training
                outs: "list[torch.Tensor]" = self.backbone(x)
                logits: "torch.Tensor" = outs[-1]
                all_outs.append(outs)
                y = F.adaptive_max_pool3d(y, logits.shape[-3:])
                target = torch.cat([target, y], dim=0)
            else:
                with torch.no_grad():
                    outs = self.backbone(x)
                    logits = outs[-1]
                    all_outs.append(logits)
                    if has_target:
                        y = F.adaptive_max_pool3d(y, logits.shape[-3:])
                        target = torch.cat([target, y], dim=0)

        if self.training:
            outs = [
                torch.cat([out[i] for out in all_outs]) for i in range(len(all_outs[0]))
            ]
            logits = outs[-1]
            loss_contributions: "torch.Tensor" = self.loss_contributions.to(device)
            loss = loss_contributions[-1] * self.loss_fn(logits, target)
            x_aux: "torch.Tensor" = F.max_pool3d(
                logits, **self.cfg.max_loss_pooling_args
            )
            y_aux: "torch.Tensor" = F.max_pool3d(
                target, **self.cfg.max_loss_pooling_args
            )
            loss += loss_contributions[-2] * self.max_loss_fn(x_aux, y_aux)

            if self.cfg.deep_supervision:
                x_aux = outs[-2]
                y_aux = F.avg_pool3d(target, **self.cfg.avg_loss_pooling_args)
                loss += loss_contributions[-3] * self.avg_loss_fn(x_aux, y_aux)

            if self.cfg.weighted_loss:
                loss /= loss_contributions.sum()

        else:
            with torch.no_grad():
                # For inference, we concatenate all outputs
                logits = torch.cat(all_outs, dim=0)
                if has_target:
                    loss = self.loss_fn(logits, target)

        if has_target:
            outputs["loss"] = loss
            outputs["dice"] = self.dice_fn(logits, target).squeeze().mean(dim=0)

        if not self.training:
            outputs["logits"] = logits
            if "location" in batch:
                outputs["location"] = batch["location"]
            if "scale" in batch:
                outputs["scale"] = batch["scale"]
            if "id" in batch:
                outputs["id"] = batch["id"]
        return outputs
