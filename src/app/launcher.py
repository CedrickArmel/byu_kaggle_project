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

# mypy: disable-error-code="misc, assignment"

import datetime
import os

import hydra
from models import LNet
from omegaconf import DictConfig
from trainers import get_lightning_trainer
from utils import get_callbacks, get_data, get_data_loader, set_seed


@hydra.main(config_path="./config", config_name="config")
def main(cfg: "DictConfig") -> "None":
    set_seed(cfg.seed)
    train_df, val_df = get_data(cfg, mode="fit")
    train_loader = get_data_loader(cfg, train_df, mode="train")
    val_loader = get_data_loader(cfg, val_df, mode="validation")
    chckpt_cb, lr_cb = get_callbacks(cfg)

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.default_root_dir = os.path.join(
        cfg.output_dir,
        cfg.backbone,
        f"seed_{cfg.seed}",
        f"fold{cfg.fold}",
        f"{start_time}",
    )

    cfg.backbone_args = dict(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.n_classes,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
    )
    trainer = get_lightning_trainer(cfg, [chckpt_cb, lr_cb])
    model = LNet(cfg)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    main()
