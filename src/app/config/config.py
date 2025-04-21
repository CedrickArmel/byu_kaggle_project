from types import SimpleNamespace
import numpy as np
import monai.transforms as mt

cfg = SimpleNamespace(**{})

# PARAMETERS
# Notebook parameters
# cfg.shuffle = False
# cfg.drop_last = False
# cfg.name = 'resnet34'
# cfg.backbone = 'resnet34'
# cfg.backbone_args = None
# cfg.epochs_step = None
# cfg.warmup = 10
# cfg.milestones = None
# cfg.end_lambda = 0.1
# cfg.schedule = "linear"
# cfg.tpu = False

## Checkpointing
cfg.checkpoint = None
cfg.load_state = False
cfg.save_checkpoint = True
cfg.resume_from = None
cfg.save_weights_only = False

## Dataset/DataLoader
cfg.batch_size = 8
cfg.batch_size_val = 2
cfg.drop_last = False
cfg.pin_memory = True
cfg.val_pin_memory = True
cfg.test_pin_memory = True
cfg.num_workers = 8
cfg.shuffle = False
cfg.train_sub_epochs = 1
cfg.val_sub_epochs = 1

## Metrics
cfg.dt_multiplier = 1
cfg.max_th = 0.5
cfg.motor_radius = 500
cfg.score_beta = 2

# model
cfg.backbone = 'resnet34'
cfg.class_weights = np.array([256, 1])
cfg.in_channels = 1
cfg.mixup_p = 1.
cfg.mixup_beta = 1.
cfg.n_classes = 1
cfg.pretrained = True
cfg.virt_sub_batch_size = -1  # split N, C, D, H, W from the loader in sub_batchs
cfg.virt_eval_sub_batch_size = 32
cfg.backbone_args = dict(spatial_dims=3,    
                         in_channels=cfg.in_channels,
                         out_channels=cfg.n_classes,
                         backbone=cfg.backbone,
                         pretrained=cfg.pretrained)
cfg.lvl_weights = np.array([0, 0, 0, 1])

# Optimisation
cfg.clip_grad = 1.0
cfg.downcast_bf16 = True
cfg.grad_accumulation = 1.
cfg.grad_norm_type = 2
cfg.mixed_precision = "bf16" 

## Optimizer
cfg.lr = 1e-5
cfg.optimizer = "SGD"
cfg.sgd_momentum = 0.8
cfg.sgd_nesterov = True,
cfg.weight_decay = 0.

# Paths
cfg.accelerate_dir = "./accelerate"
cfg.data_folder = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
cfg.df_path = "/kaggle/input/drx75c-byu-ds01-folded-tomograms/byu_folded_tomograms_seed57.csv"
cfg.name = 'resnet34'
cfg.output_dir = f"./{cfg.name}"
cfg.submission_dir = "/kaggle/working"

## Scheduler
cfg.end_lambda = 1e-3
cfg.milestones = 10
cfg.schedule ="multistep"
cfg.warmup = 10  # Warmup steps, not epochs

## Tracking
cfg.track_grad_norm = True
cfg.track_weight_norm = True

## Training
cfg.best_th = 0.5
cfg.chkpt_step = 100
cfg.compile_model = True
cfg.epochs = 10
cfg.epochs_step = None
cfg.eval_steps = 50
cfg.fold = 0
cfg.last_score = 0.0
cfg.milestones = None
cfg.seed = 57

## PP
cfg.nms_radius = 16

## TRANSFORMS
cfg.gamma = 2
cfg.new_size = (300, 500, 500)
cfg.roi_size = [96, 96, 96]
cfg.sub_batch_size = 2

# Transforms: Train & Val
cfg.static_transforms = mt.Compose([
    mt.EnsureChannelFirstd(keys=["input", "target"], channel_dim="no_channel"),
    mt.AdjustContrastd(keys=["input"], gamma=cfg.gamma),
    mt.ScaleIntensityd(keys=["input"]),
    mt.Orientationd(keys=["input", "target"], axcodes="RAS"),
])

# Transforms: Train only
cfg.train_transforms = mt.Compose([
    mt.RandCropByLabelClassesd(
        keys=["input", "target"], label_key="target",
        spatial_size=cfg.roi_size, num_samples=cfg.sub_batch_size, num_classes=2, ratios=[1,1],
        warn=False),
    mt.RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=2),
    mt.RandRotate90d(keys=["input", "target"],prob=0.75,max_k=3, spatial_axes=(0, 1)),
    mt.RandRotated(keys=["input", "target"], prob=0.5,range_x=0.78,range_y=0.,range_z=0., padding_mode='reflection'),
])

# Transforms: Val only
cfg.eval_transforms = mt.Compose([mt.GridPatchd(keys=["input", "target"],patch_size=cfg.roi_size, pad_mode='reflect')])

# Transforms: Test only
cfg.test_transforms = mt.Compose([
    mt.EnsureChannelFirstd(keys=["input"], channel_dim="no_channel"),
    mt.AdjustContrastd(keys=["input"], gamma=cfg.gamma),
    mt.ScaleIntensityd(keys=["input"]),
    mt.Orientationd(keys=["input"], axcodes="RAS"),
    mt.GridPatchd(keys=["input"], patch_size=cfg.roi_size, pad_mode='reflect')
])
