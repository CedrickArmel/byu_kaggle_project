import datetime
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
from torchinfo import summary
from app.metrics import BYUFbeta
from app.metrics.metrics import get_topk_by_id, thresholder, filter_negatives
from app.models.lightning import Net
from app.models import LNet
from app.utils import get_data, get_data_loader
from app.processings.post_processing import get_output_size, reconstruct, simple_nms

from numpy.typing import NDArray
from scipy.spatial import KDTree
from torchmetrics.utilities import dim_zero_cat

OmegaConf.register_new_resolver("eval", resolver=eval, replace=True)

if __name__ == "__main__":
    cfg = OmegaConf.load("src/app/config/config.yaml")
    cfg.backbone = "resnet10"
    cfg.val_persistent_workers = True
    train_df, val_df = get_data(cfg, mode="fit")
    train_loader = get_data_loader(cfg, train_df, mode="train")
    val_loader = get_data_loader(cfg, val_df, mode="validation")

    state =  torch.load("/kaggle/working/resnet10/version_33/checkpoints/last.ckpt", map_location="cpu")
    cfg.virt_eval_sub_batch_size = 1

    model = LNet(cfg)
    model.load_state_dict(state["state_dict"])
    tr_it = iter(val_loader)
    counter = 0
    for batch in range(len(val_loader)):
        batch = next(tr_it)
        if batch["id"][0] not in [16, 38, 200, 422]:
            continue
        print(batch["id"][0])
        b_cuda = {k:v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model.to("cuda:0")
        model.eval()
        with torch.no_grad():
            output = model(b_cuda)
        logits = output["logits"]
        print(output["dice"])
        logits = logits.cpu().numpy()
        torch.save(logits, f"/kaggle/working/logits{batch['id'][0]}.pt")
        counter += 1
        if counter == 4:
            break