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

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def get_output_size(
    img: "torch.Tensor",
    locations: "torch.Tensor",
    roi_size: "torch.Tensor",
    device: "torch.device",
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
    output_size = torch.zeros(5, device=device)
    s: "torch.Tensor" = torch.unique(shapes, dim=0).squeeze().to(device)
    # s = [s[i] + roi_size[i] for i in range(len(s))]
    s = s + roi_size
    output_size[0] = shapes.shape[0]
    output_size[1] = img.shape[1]
    output_size[2:] = s
    return output_size.to(torch.int)


def reconstruct(
    img: "torch.Tensor",
    locations: "torch.Tensor",
    out_size: "torch.Tensor",
    crop_size: "torch.Tensor",
    device: "torch.device",
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
    reconstructed_img = torch.zeros(out_size.tolist(), device=device)
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
    cfg: "DictConfig", net_output: "dict[str, Any]"
) -> "torch.Tensor":
    """Post-process the output of the model to get the final coordinates and confidence scores.
    Args:
        cfg (DictConfig): Configuration object containing project's parameters.
        net_output (dict): The output of the model.
    Returns:
        torch.Tensor: The final coordinates and confidence scores.
    """
    device = net_output["logits"].device
    new_size = torch.tensor(cfg.new_size, device=net_output["logits"].device)
    roi_size = torch.tensor(cfg.roi_size, device=net_output["logits"].device)

    img: "torch.Tensor" = net_output["logits"].detach()

    locations: "torch.Tensor" = net_output["location"]
    scales: "torch.Tensor" = net_output["scale"]
    tomo_ids: "torch.Tensor" = torch.tensor(net_output["id"], device=device)

    # TODO: in future interpolate only if the size is not equal to roi_size there

    out_size = get_output_size(img, locations, roi_size, device)
    rec_img = reconstruct(
        img=img,
        locations=locations,
        out_size=out_size,
        crop_size=roi_size,
        device=device,
    )

    s = torch.tensor(rec_img.shape[-3:], device=device)
    delta = (s - new_size) // 2  # delta to remove padding added during transforms
    dz, dy, dx = delta.tolist()
    nz, ny, nx = new_size.tolist()

    rec_img = rec_img[:, :, dz : nz + dz, dy : ny + dy, dx : nx + dx]

    rec_img = F.interpolate(
        rec_img,
        size=[d // 2 for d in new_size.tolist()],
        mode=cfg.down_interp_mode,
        align_corners=cfg.down_align_corners,
    )

    preds: "torch.Tensor" = rec_img.softmax(1)
    preds = preds[:, 1, :][None,]

    nms: "torch.Tensor" = simple_nms(preds, nms_radius=cfg.nms_radius)  # (1,B, D, H, W)
    nms = nms.squeeze(dim=0)  # (B, D, H, W)

    flat_nms = nms.reshape(nms.shape[0], -1)  # (B, D*H*W)
    conf, indices = torch.topk(flat_nms, k=cfg.topk, dim=1)
    zyx = torch.stack(torch.unravel_index(indices, nms.shape[-3:]), dim=-1)  # (B, K, 3)

    b = torch.arange(zyx.shape[0], device=device).unsqueeze(1)
    ids = torch.unique(tomo_ids.reshape(zyx.shape[0], -1), dim=1).expand(
        zyx.shape[0], cfg.topk
    )

    zyx = ((zyx * 2) / scales[b]).round().to(torch.int)
    conf = conf.to(torch.float32)

    ids = ids.reshape(-1, 1)
    conf = conf.reshape(-1, 1)
    zyx = zyx.reshape(-1, 3)

    output: "torch.Tensor" = torch.cat([zyx, ids, conf], dim=1)
    return output
