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

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


def get_output_size(
    img: "torch.Tensor", locations: "torch.Tensor", roi_size: "list[int]"
) -> "torch.Tensor":
    """Get the output size of the reconstructed image.
    Args:
        img (torch.Tensor): The input image tensor.
        locations (torch.Tensor): The locations of the detected points.
        roi_size (list): The size of the region of interest.
    Returns:
        torch.Tensor: The output size of the reconstructed image.
    """
    shapes = locations.max(2)[0]
    output_size = torch.zeros(5)
    s = torch.unique(shapes, dim=0).squeeze()
    s = [s[i] + roi_size[i] for i in range(len(s))]
    output_size[0] = shapes.shape[0]
    output_size[1] = img.shape[1]
    output_size[2:] = torch.tensor(s)
    return output_size.to(torch.int)


def reconstruct(
    img: "torch.Tensor",
    locations: "torch.Tensor",
    out_size: "torch.Tensor",
    crop_size: "list[int]",
) -> "torch.Tensor":
    """Reconstruct the image from the detected points.
    Args:
        img (torch.Tensor): The input image tensor.
        locations (torch.Tensor): The locations of the detected points.
        out_size (torch.Tensor): The output size of the reconstructed image.
        crop_size (list): The size of the region of interest.
    Returns:
        torch.Tensor: The reconstructed image tensor.
    """
    reconstructed_img = torch.zeros(tuple(out_size), device=img.device)
    reshape = list([locations.shape[0], locations.shape[2]]) + list(img.shape[1:])
    image = img.reshape(reshape)
    for i in range(out_size[0]):
        for j in range(locations.shape[2]):
            reconstructed_img[i][
                :,
                locations[i][0][j] : locations[i][0][j] + crop_size[0],
                locations[i][1][j] : locations[i][1][j] + crop_size[1],
                locations[i][2][j] : locations[i][2][j] + crop_size[2],
            ] = image[i][j, :]
    return reconstructed_img


def simple_nms(scores: "torch.Tensor", nms_radius: "int") -> "torch.Tensor":
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x: "torch.Tensor") -> "torch.Tensor":
        """Max pooling operation to find the maximum value in a local neighborhood."""
        return F.max_pool3d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)


def post_process_pipeline(
    cfg: "SimpleNamespace", net_output: "dict[str, Any]"
) -> "torch.Tensor":
    """Post-process the output of the model to get the final coordinates and confidence scores.
    Args:
        cfg (types.SimpleNamespace): Configuration object containing project's parameters.
        net_output (dict): The output of the model.
    Returns:
        torch.Tensor: The final coordinates and confidence scores.
    """
    img: "torch.Tensor" = net_output["logits"].detach()
    device: "torch.device" = img.device

    locations: "torch.Tensor" = net_output["location"].cpu()
    scales: "torch.Tensor" = net_output["scale"]

    tomo_ids: "list[int]" = [
        net_output["id"][i] for i in range(0, len(net_output["id"]), locations.shape[2])
    ]
    dims_: "list[torch.Size]" = [
        net_output["dims"][i]
        for i in range(0, len(net_output["dims"]), locations.shape[2])
    ]

    img = F.interpolate(
        img,
        size=(cfg.roi_size[0], cfg.roi_size[1], cfg.roi_size[2]),
        mode="trilinear",
        align_corners=False,
    )

    out_size = get_output_size(img, locations, cfg.roi_size)

    rec_img = reconstruct(img, locations, out_size=out_size, crop_size=cfg.roi_size)

    s = rec_img.shape[-3:]
    rec_img = F.interpolate(
        rec_img,
        size=(s[0] // 2, s[1] // 2, s[2] // 2),
        mode="trilinear",
        align_corners=False,
    )

    preds = rec_img.softmax(1)

    p1: "torch.Tensor" = preds[:, 0, :][None,]
    y: "torch.Tensor" = simple_nms(p1, nms_radius=cfg.nms_radius)
    kps: "tuple[torch.Tensor, ...]" = torch.where(y.squeeze() > 0)
    bzyx: "torch.Tensor" = torch.stack(kps, -1)
    conf: "torch.Tensor" = y.squeeze()[kps]

    delta = ((torch.tensor(s) - torch.tensor(cfg.new_size)) // 2).to(device)
    dims = torch.tensor(dims_, device=device, dtype=torch.int)
    b = bzyx[:, 0].to(torch.long)
    zyx = bzyx[:, 1:]
    zyx = (
        (((zyx * 2) - delta) / scales[b]).round().to(torch.int)
    )  # delta to remove padding added during transforms
    z, y, x = zyx[:, 0], zyx[:, 1], zyx[:, 2]
    ids_: "list[int]" = [tomo_ids[int(bb)] for bb in b]
    conf = conf.to(torch.float32)

    dims_tensor: "torch.Tensor" = dims[b]
    in_bounds = (
        (z > 0)
        & (z < dims_tensor[:, 0])
        & (y > 0)
        & (y < dims_tensor[:, 1])
        & (x > 0)
        & (x < dims_tensor[:, 2])
        & (conf > 0.01)
    )

    z = z[in_bounds]
    y = y[in_bounds]
    x = x[in_bounds]
    conf = conf[in_bounds]
    ids_ = [ids_[i] for i in in_bounds.nonzero(as_tuple=True)[0]]
    empty_tomos = [tid for tid in tomo_ids if tid not in set(ids_)]

    if len(empty_tomos) > 0:
        dummy_coords = torch.full(
            (len(empty_tomos),), -1, dtype=torch.int, device=device
        )
        dummy_conf = torch.zeros(len(empty_tomos), dtype=torch.float32, device=device)
        z = torch.cat([z, dummy_coords])
        y = torch.cat([y, dummy_coords])
        x = torch.cat([x, dummy_coords])
        conf = torch.cat([conf, dummy_conf])
        ids_.extend(empty_tomos)
    ids: "torch.Tensor" = torch.tensor(ids_, device=device)
    output: "torch.Tensor" = torch.stack([z, y, x, conf, ids], dim=1)
    return output
