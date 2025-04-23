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

import os
from glob import glob
from types import SimpleNamespace
from typing import Any

import monai.transforms as mt
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, read_file


def load_image(path: "str") -> torch.Tensor:
    """Load an image from a file path."""
    img = read_file(path)
    img = decode_jpeg(img, mode="GRAY")
    return img


class BYUCustomDataset(Dataset):  # type: ignore[misc]
    """Custom dataset for BYU data."""

    def __init__(
        self,
        cfg: "SimpleNamespace",
        mode: "str" = "train",
        aug: "mt.Transform | None" = None,
        df: "pd.DataFrame | None" = None,
    ) -> None:
        """_BYUCustomDataset_

        Args:
            cfg (SimpleNamespace): Configuration object containing project's parameters.
            mode (str, optional): Whether loading the data for train/validation/test. Defaults to "train".
            aug (mt.Transform, optional): Transformation to apply to the data. Defaults to None.
            df (pd.DataFrame|None, optional): Dataframe containing the fold. Defaults to None.

        Raises:
            ValueError: If the mode argument is not one of train, validation or test.
        """
        if mode not in ["train", "validation", "test"]:
            raise ValueError("mode argument must be one of train, validation or test!")

        if mode != "test" and df is None:
            raise ValueError(
                "df argument must be provided for train and validation modes!"
            )

        self.cfg = cfg
        self.mode = mode
        self.data_folder = cfg.data_folder

        if self.mode != "test":
            self.transforms: "mt.Transform | None" = aug
            self.static_transforms: "mt.Transform" = cfg.static_transforms
        else:
            self.test_transforms: "mt.Transform" = cfg.test_transforms

        self.new_size: "tuple[int]" = cfg.new_size

        if self.mode != "test":
            self.df: "pd.DataFrame" = df
            self.tomo_list: "list[str]" = sorted(self.df.tomo_id.unique().tolist())
            self.tomo_dict = self.df.groupby("tomo_id")
        else:
            self.tomo_list = sorted(
                [
                    path.split("/")[-1]
                    for path in glob(os.path.join(cfg.data_folder, "test", "**"))
                ]
            )
        self.tomo_mapping = pd.DataFrame(
            {"tomo_id": self.tomo_list, "id": range(len(self.tomo_list))}
        )

        if self.mode == "train":
            self.sub_epochs: "int" = cfg.train_sub_epochs
            self.len = len(self.tomo_list) * self.sub_epochs
        else:
            self.sub_epochs = 1
            self.len = len(self.tomo_list)

    def __getitem__(self, idx: "int") -> "dict[str, Any]":
        """_Return the data for the given index_"""
        data_: "dict[str, Any]" = self.get_data_dict(idx)
        if self.mode != "test":
            data_ = self.static_transforms(data_)
            data: "list[dict[str, Any]] | dict[str, Any]" = self.transforms(data_)  # type: ignore[misc]
            if self.mode == "train":
                feature_dict = {
                    "input": torch.stack([item["input"].as_tensor() for item in data]),  # type: ignore[union-attr, index]
                    "target": torch.stack(
                        [item["target"].as_tensor() for item in data]  # type: ignore[union-attr, index]
                    ),
                    "id": data[0]["id"],  # type: ignore[index]
                }
            else:
                feature_dict = {
                    "input": data["input"].as_tensor(),  # type: ignore[call-overload]
                    "target": data["target"].as_tensor(),  # type: ignore[call-overload]
                    "location": torch.from_numpy(data["input"].meta["location"]),  # type: ignore[call-overload]
                    "id": data["id"],  # type: ignore[call-overload]
                    "scale": data["scale"],  # type: ignore[call-overload]
                    "dims": data["dim"],  # type: ignore[call-overload]
                }
        else:
            data = self.test_transforms(data_)
            feature_dict = {
                "input": data["input"].as_tensor(),  # type: ignore[call-overload]
                "location": torch.from_numpy(data["input"].meta["location"]),  # type: ignore[call-overload]
                "id": data["id"],  # type: ignore[call-overload]
                "scale": data["scale"],  # type: ignore[call-overload]
                "dims": data["dim"],  # type: ignore[call-overload]
            }
        return feature_dict

    def __len__(self) -> "int":
        """_Return the length of the dataset_"""
        return self.len

    def load_tomogram(self, tomo_id: "str") -> "torch.Tensor":
        """_Return one tomogram_

        Args:
            tomo_id (str): Tomogram to be loaded' ID

        Returns:
            torch.Tensor: _Loaded tomogram_
        """
        mode = "train" if self.mode == "validation" else self.mode
        slices_path = os.path.join(self.data_folder, mode, tomo_id, "*.jpg")
        image = torch.stack(
            [load_image(path) for path in sorted(glob(slices_path, recursive=True))]
        ).squeeze()
        return image

    def get_locs_n_vxs(
        self, tomo_id: "str", scale: "torch.Tensor"
    ) -> "torch.Tensor | None":
        "return motors centers coordinates (zyx) and voxel spacing (vxs)"
        zyx_: "NDArray" = self.tomo_dict.get_group(tomo_id)[
            ["z", "y", "x"]
        ].values.astype(int)
        if (zyx_ == -1).any():
            return None
        zyx: "torch.Tensor" = torch.ceil(torch.tensor(zyx_) * scale)
        # vxs = self.tomo_dict.get_group(tomo_id).iloc[0]["vxs"]
        return zyx.to(torch.int)  # vxs

    def reduce_volume(self, volume: "torch.Tensor") -> "torch.Tensor":
        """_Downsample to a smaller tomogram_

        Args:
            volume (torch.Tensor): The Volume to be downsampled.

        Returns:
            torch.Tensor: Downsampled volume.Ò
        """
        input_volume = volume.unsqueeze(dim=0).unsqueeze(dim=0)
        output: "torch.Tensor" = F.interpolate(
            input_volume, size=self.new_size, mode="nearest"
        ).squeeze()
        return output

    def compute_scale(self, img_size: "torch.Size") -> "torch.Tensor":
        """_Compute the scale factor to downsample the tomogram_
        Args:
            img_size (torch.Size): The size of the tomogram.
        Returns:
            torch.Tensor: The scale factor to downsample the tomogram.
        """
        D, H, W = img_size
        scale = torch.tensor(self.new_size) / torch.tensor([D, H, W])
        return scale

    def get_data_dict(self, idx: "int") -> "dict[str, Any]":
        """_Return the data dictionary for the given index_
        Args:
            idx (int): The index of the data to be returned.
        Returns:
            dict: The data dictionary containing the input, target, id, scale and dim.
        """
        tomo_id: "str" = self.tomo_list[idx // self.sub_epochs]
        num_tomo_idx: "int" = self.tomo_mapping.loc[
            self.tomo_mapping.tomo_id == tomo_id, "idx"
        ].iloc[0]
        tomogram: "torch.Tensor" = self.load_tomogram(tomo_id)
        s: "torch.Size" = tomogram.shape
        scale: "torch.Tensor" = self.compute_scale(s)
        tomogram = self.reduce_volume(tomogram)

        if self.mode != "test":
            zyx: "torch.Tensor | None" = self.get_locs_n_vxs(
                tomo_id, scale
            )  # could return vxs there. see get_locs_n_vxs definition
            mask = torch.zeros_like(tomogram)
            if zyx is not None:
                mask[zyx[:, 0], zyx[:, 1], zyx[:, 2]] = 1.0
            return {
                "input": tomogram,
                "target": mask,
                "id": num_tomo_idx,
                "scale": scale,
                "dim": s,
            }
        else:
            return {"input": tomogram, "id": num_tomo_idx, "scale": scale, "dim": s}
