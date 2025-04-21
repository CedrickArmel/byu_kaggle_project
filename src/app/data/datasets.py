import os
from glob import glob

import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.io import read_file, decode_jpeg


def load_image(path):
    img = read_file(path)
    img =  decode_jpeg(img, mode="GRAY")
    return img


class BYUCustomDataset(Dataset):
    """Custom dataset for BYU data.
    """

    def __init__(self, cfg, mode="train", aug=None, df=None):
        if mode not in ["train", "validation", "test"]:
            raise ValueError("mode argument must be one of train, validation or test!")
            
        self.cfg = cfg
        self.mode = mode
        self.data_folder = cfg.data_folder
        
        if self.mode != "test":
            self.transforms = aug
            self.static_transforms = cfg.static_transforms
        else:
            self.test_transforms = cfg.test_transforms
        
        self.new_size = cfg.new_size
        
        if self.mode != "test":
            self.df = df
            self.tomo_list = sorted(self.df.tomo_id.unique().tolist())
            self.tomo_dict = self.df.groupby('tomo_id')
        else:
            self.tomo_list = sorted([path.split("/")[-1] for path 
                                     in glob(os.path.join(cfg.data_folder, "test", "**"))])
        self.tomo_mapping = pd.DataFrame({"tomo_id": self.tomo_list, "id": range(len(self.tomo_list))})
        
        if self.mode == "train":
            self.sub_epochs = cfg.train_sub_epochs
            self.len = len(self.tomo_list) * self.sub_epochs
        else:
            self.sub_epochs = 1
            self.len = len(self.tomo_list)

    def __getitem__(self, idx):
        data = self.get_data_dict(idx)
        if self.mode != "test":
            data = self.static_transforms(data)
            data = self.transforms(data)
            if self.mode == "train":
                feature_dict = {
                    "input": torch.stack([item['input'] for item in data]).as_tensor(),
                    "target": torch.stack([item['target'] for item in data]).as_tensor(),
                    "id": data[0]["id"]}
            else:
                feature_dict = {
                    "input": data['input'].as_tensor(),
                    "target": data['target'].as_tensor(),
                    "location": torch.from_numpy(data['input'].meta['location']),
                    "id": data["id"],
                    "scale": data["scale"],
                    "dims": data["dim"]
                }
        else:
            data = self.test_transforms(data)
            feature_dict = {
                "input": data['input'].as_tensor(),
                "location": torch.from_numpy(data['input'].meta['location']),
                "id": data["id"],
                "scale": data["scale"],
                "dims": data["dim"]
                }
        return feature_dict
        
    def __len__(self):
        return self.len

    def load_tomogram(self, tomo_id):
        mode = "train" if self.mode == "validation" else self.mode
        slices_path = os.path.join(self.data_folder, mode, tomo_id, "*.jpg")
        image = torch.stack([load_image(path) for path in sorted(glob(slices_path, recursive=True))]).squeeze()
        return image

    def get_locs_n_vxs(self, tomo_id, scale):
        "return motors centers coordinates (zyx) and voxel spacing (vxs)"
        zyx = self.tomo_dict.get_group(tomo_id)[["z", "y", "x"]].values.astype(int)
        if (zyx == -1).any():
            return None
        zyx = torch.ceil(torch.tensor(zyx) * scale)
        # vxs = self.tomo_dict.get_group(tomo_id).iloc[0]["vxs"]
        return zyx.to(torch.int)  # vxs

    def reduce_volume(self, volume):
        input_volume = volume.unsqueeze(dim=0).unsqueeze(dim=0)
        output = F.interpolate(input_volume, size=self.new_size, mode="nearest").squeeze()
        return output

    def compute_scale(self, img_size):
        D, H, W = img_size
        scale = torch.tensor(self.new_size) / torch.tensor([D, H, W])
        return scale

    def get_data_dict(self, idx):
        tomo_id = self.tomo_list[idx // self.sub_epochs]
        num_tomo_idx = self.tomo_mapping.loc[self.tomo_mapping.tomo_id == tomo_id, 'idx'].iloc[0]
        tomogram = self.load_tomogram(tomo_id)
        s = tomogram.shape
        scale = self.compute_scale(s)
        tomogram = self.reduce_volume(tomogram)
        
        if self.mode != "test":
            zyx = self.get_locs_n_vxs(tomo_id, scale)  # could return vxs there. see get_locs_n_vxs definition
            mask = torch.zeros_like(tomogram)
            if zyx is not None:
                mask[zyx[:, 0], zyx[:, 1], zyx[:, 2]] = 1.
            return {"input": tomogram,
                    "target": mask,
                    "id": num_tomo_idx,
                    "scale": scale,
                    "dim": s
                   }
        else:
            return {"input": tomogram, "id": num_tomo_idx, "scale": scale, "dim": s}


