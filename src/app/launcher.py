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
import sys

from models import LNet
from omegaconf import OmegaConf
from omegaconf.OmegaConf import DictConfig
from trainers import lightning_trainer
from utils import get_callbacks, get_data, get_data_loaders

# from config import cfg
from data import get_transforms


def main(cfg: "DictConfig") -> "None":
    cfg.train_df, cfg.val_df = get_data(cfg)
    (
        cfg.static_transforms,
        cfg.train_transforms,
        cfg.eval_transforms,
        cfg.test_transforms,
    ) = get_transforms(cfg)
    train_loader, val_loader = get_data_loaders(cfg)
    chckpt_cb, lr_cb = get_callbacks(cfg)

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.default_root_dir = os.path.join(
        "/kaggle/working/{cfg.backbone}",
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
    trainer = lightning_trainer(cfg, [chckpt_cb, lr_cb])
    model = LNet(cfg)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=None,
    )


if __name__ == "__main__":
    cfg: "DictConfig" = OmegaConf.load("src/app/config/config.yaml")
    cli_overrides: "DictConfig" = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, cli_overrides)
    main(cfg)
