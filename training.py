import gc
import os
import json
import time
import traceback
import cv2
import numpy as np
import torch as T
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import lovely_tensors as lt
# import torchvision.models as torch_models

from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.tuner import tuning

from motion_capture.data.datasets import WIDERFaceDataset, WFLWDataset, COFWColorDataset, MPIIDataset, COCO2017PersonKeypointsDataset, COCO2017PanopticsDataset, COCO2017WholeBodyDataset
from motion_capture.data.preprocessing import ImagePertubators
from motion_capture.model.models import UpsampleCrossAttentionNetwork, find_best_checkpoint_path
from motion_capture.data.datamodules import BboxDataModule


# ---------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    EXPERIMENT_NAME = "bbox_backbone"
    RUN_NAME = "version_1"
    MODEL_CHECKPOINTS_PATH = "checkpoints"
    RANDOM_SEED = 1
    
    DEVICE = T.device("cuda:0")
    CONTINUE_TRAINING = True
    NUM_TRAIN_WORKERS = 8
    NUM_VAL_WORKERS = 4
    IMAGE_SHAPE = (224, 224) # Width x Height
    BATCH_SIZE = 80
    TRAIN_VAL_TEST_SPLIT = [0.8, 0.15, 0.05]
    
    OUTPUT_SIZE = 4
    OUTPUT_LENGTH = 32
    
    # general params
    LOGGER_ARGS = {
        "save_dir": "logs/",
        "tracking_uri": None,
        "tags": ["bbox", "backbone"],
    }
    
    CHEKCPOINT_CALLBACK_ARGS = {
        "save_top_k": 3,
        "every_n_epochs": 1,
        "monitor": "val_loss",
        "filename": "{epoch}-{step}-{val_loss:.4f}",
        "mode": "min",
    }
    
    MODEL_STRUCTURE_ARGS = {
        "output_size": OUTPUT_SIZE,
        "output_length": OUTPUT_LENGTH,
        "backbone_output_size": 512,
        "neck_output_size": 256,
        "head_latent_size": 256,
    }
    
    MODEL_TRAIN_ARGS = {
        "loss_fn": T.nn.functional.l1_loss,
        "optimizer": T.optim.Adam,
        "optimizer_kwargs": {
            "lr": 1e-04,
            "weight_decay": 0.005,
            "momentum": 0.9,
            "rho": 0.5,
        },
        "lr_scheduler": T.optim.lr_scheduler.CosineAnnealingLR,
        "lr_scheduler_kwargs": {
            "T_max": 10,
            "eta_min": 1e-06,
            "last_epoch": -1,
        }
    }
    
    TRAINER_ARGS = {
        "max_epochs": 200,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 1,
        "accumulate_grad_batches": 5,
        "gradient_clip_algorithm": "norm",
        "gradient_clip_val": 0.2
    }
    
    # stochastic weight averaging
    SWA_ARGS = {
        "swa_lrs": [1e-04, 1e-05],
        "swa_epoch_start": 5,
        "annealing_epochs": 10,
        "annealing_strategy": "cosine",
    }
    
    
    DATASET_ARGS = {
        "datase_classt": WFLWDataset,
        "dataset_kwargs": {
            "output_full_image_shape_WH": IMAGE_SHAPE,
            "output_face_image_shape_WH": (112, 112),
            "max_number_of_faces": OUTPUT_LENGTH,
            "image_path": "//192.168.2.206/data/datasets/WFLW/images",
            "annotation_path": "//192.168.2.206/data/datasets/WFLW/annotations",
            "image_pertubator": ImagePertubators.BASIC(),
            "padding": "random_elements"
        },
        "image_key": "fullImage",
        "bbox_key": "faceBboxes"
    }
    # dataset = COCO2017PanopticsDataset(
    #     image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
    #     panoptics_path="//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations",
    #     output_image_shape_WH=IMAGE_SHAPE,
    #     instance_images_output_shape_WH=(8, 8),
    #     max_number_of_instances=OUTPUT_LENGTH,
    #     load_segmentation_masks=False,
    #     image_pertubator=ImagePertubators.BASIC(),
    #     # limit_to_first_n = 320
    # )
    # dataset = WIDERFaceDataset(
    #     output_image_shape_WH=IMAGE_SHAPE, 
    #     max_number_of_faces=OUTPUT_LENGTH,
    #     train_path="//192.168.2.206/data/datasets/WIDER-Face/train",
    #     val_path="//192.168.2.206/data/datasets/WIDER-Face/val",
    #     image_pertubator = ImagePertubators.BASIC(),
    #     center_bbox = True
    # )
    
    
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing ...")
    
    pl.seed_everything(RANDOM_SEED)
    
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing trainer ...")
    
    current_checkpoint_path = os.path.join(MODEL_CHECKPOINTS_PATH, EXPERIMENT_NAME, RUN_NAME)
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        **LOGGER_ARGS
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath = current_checkpoint_path,
        verbose = True,
        **CHEKCPOINT_CALLBACK_ARGS
    )
    swa = StochasticWeightAveraging(**SWA_ARGS, device=DEVICE)
    
    trainer = pl.Trainer(
        accelerator = DEVICE,
        logger=logger,
        callbacks = [checkpoint_callback, swa],
        **TRAINER_ARGS,
        # fast_dev_run=True,
    )
    
    # ---------------------------------------------------------------------------------------------------------------
    print("initialize model ...")
    
    if CONTINUE_TRAINING:
        best_ckpt_pth = find_best_checkpoint_path(current_checkpoint_path)
        model = UpsampleCrossAttentionNetwork.load_from_checkpoint(best_ckpt_pth)
    else:
        model = UpsampleCrossAttentionNetwork(**MODEL_STRUCTURE_ARGS, **MODEL_TRAIN_ARGS)
    
    # ---------------------------------------------------------------------------------------------------------------
    print("initialize dataset ...")
    
    data_module = BboxDataModule(
        **DATASET_ARGS,
        image_shape=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        num_train_workers=NUM_TRAIN_WORKERS,
        num_val_workers=NUM_VAL_WORKERS
    )
    
    # ---------------------------------------------------------------------------------------------------------------
    # print("find lr ...")
    # tuner = tuning.Tuner(trainer=trainer)
    # lr_finder = tuner.lr_find(
    #     model=model, 
    #     train_dataloaders=train_dataloader, 
    #     val_dataloaders=val_dataloader,
    #     num_training=50
    # )
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # input("Press Enter to continue ...")
    
    # ---------------------------------------------------------------------------------------------------------------
    print("fitting model ...")
    
    trainer.fit(model=model, datamodule=data_module)
    