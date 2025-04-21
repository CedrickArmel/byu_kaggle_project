import torch_xla as xla

from config import cfg
from trainers import trainer
from models import Net

def _mp_fn(index, cfg, model):
  trainer(cfg, model)

if __name__ == '__main__':
    cfg.train = True
    cfg.tpu = True
    cfg.backbone = 'resnet10'
    cfg.name = 'resnet10'
    cfg.output_dir = f"./{cfg.name}"
    cfg.pretrained = True

    ## Dataset/DataLoader
    cfg.num_workers =  10
    cfg.batch_size = 10
    cfg.batch_size_val = 2
    cfg.train_sub_epochs = 1
    cfg.val_sub_epochs = 1
    cfg.prefetch_factor = 2
    cfg.pin_memory = False
    cfg.val_pin_memory = False

    # Closure
    cfg.closure_steps = 10
    cfg.async_closure = False

    ## Model
    model = Net(cfg)
    if cfg.pretrained:
       for param in model.backbone.encoder.parameters():
          param.requires_grad = False
      
    xla.launch(
       _mp_fn, args=(cfg, model))