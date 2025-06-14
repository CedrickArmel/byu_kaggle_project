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

# mypy: ignore-errors


import torch
from omegaconf import OmegaConf

from app.models import LNet
from app.utils import get_data, get_data_loader

OmegaConf.register_new_resolver("eval", resolver=eval, replace=True)

if __name__ == "__main__":
    cfg = OmegaConf.load("src/app/config/config.yaml")
    cfg.fold = 0
    cfg.backbone = "resnet10"
    cfg.backbone_args.pretrained = False
    cfg.val_persistent_workers = True
    train_df, val_df = get_data(cfg, mode="fit")
    train_loader = get_data_loader(cfg, train_df, mode="train")
    val_loader = get_data_loader(cfg, val_df, mode="validation")

    # state = torch.load(
    #    "/kaggle/working/resnet10/seed_4294967295/fold3/version_0/checkpoints/last.ckpt",
    #    map_location="cpu",
    # )
    cfg.virt_eval_sub_batch_size = 4

    model = LNet(cfg)
    # model.load_state_dict(state["state_dict"])

    tr_it = iter(val_loader)
    counter = 0
    for batch in range(len(val_loader)):
        batch = next(tr_it)
        if batch["id"][0] == 212:
            continue
        print(batch["id"][0])
        # b_cuda = {
        #    k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
        #    for k, v in batch.items()
        # }
        # model.to("cuda:0")
        model.eval()
        with torch.no_grad():
            output = model(batch)
        logits = output["logits"]
        print(output["loss"])
        logits = logits.cpu().numpy()
        print(logits.shape)
        torch.save(logits, f"/kaggle/working/logits{batch['id'][0]}.pt")
        counter += 1
        if counter == 1:
            break
