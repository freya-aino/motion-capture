import gc
import os
import json
import time
import traceback
import cv2
import importlib
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

import motion_capture.data.datasets as datasets # WIDERFaceDataset, WFLWDataset, COFWColorDataset, MPIIDataset, COCO2017PersonKeypointsDataset, COCO2017PanopticsDataset, COCO2017WholeBodyDataset
import motion_capture.data.preprocessing as preprocessing
from motion_capture.model.models import UpsampleCrossAttentionNetwork, find_best_checkpoint_path
from motion_capture.data.datamodules import BboxDataModule


# ---------------------------------------------------------------------------------------------------------------

import optuna
import hydra
from omegaconf import DictConfig, OmegaConf


def objective(cfg: dict): # , trial: optuna.Trial
    
    print("initializing logger ...")
    logger = MLFlowLogger(
        experiment_name=cfg.experimentName,
        run_name=cfg["runName"],
        **cfg.logger
    )
#     # ---------------------------------------------------------------------------------------------------------------
#     print("initializing callbacks ...")
#     current_checkpoint_path = os.path.join(cfg.modelCheckpointPath, cfg.experimentName, RUN_NAME)
#     checkpoint_callback = ModelCheckpoint(
#         dirpath = current_checkpoint_path,
#         verbose = True,
#         **cfg.callbacks.checkpoint
#     )
#     swa = StochasticWeightAveraging(device=cfg.device, **cfg.callbacks.swa)
#     # ---------------------------------------------------------------------------------------------------------------
#     print("initializing trainer ...")
#     trainer = pl.Trainer(
#         accelerator = cfg.device,
#         logger=logger,
#         callbacks = [checkpoint_callback, swa],
#         num_sanity_val_steps = 0,
#         log_every_n_steps = 1,
#         **cfg.trainer
#     )
#     # ---------------------------------------------------------------------------------------------------------------
#     print("initialize model ...")
#     if cfg["continueTraining"]:
#         best_ckpt_pth = find_best_checkpoint_path(current_checkpoint_path)
#         model = UpsampleCrossAttentionNetwork.load_from_checkpoint(best_ckpt_pth)
#     else:
#         model = UpsampleCrossAttentionNetwork(**cfg.modelStructure, **cfg.modelTraining)
#     # ---------------------------------------------------------------------------------------------------------------
#     print("initialize dataset ...")
    
#     data_module = BboxDataModule(
#         **cfg.dataset,
#         image_shape=cfg.experiments.imageShape,
#         batch_size=BATCH_SIZE,
#         train_val_test_split=TRAIN_VAL_TEST_SPLIT,
#         num_train_workers=NUM_TRAIN_WORKERS,
#         num_val_workers=NUM_VAL_WORKERS
#     )


@hydra.main(config_path="configs/hydra", config_name="experiments/bboxBackbone/run_1", version_base=None)
def run_optuna(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = cfg["experiments"]["bboxBackbone"]
    
    # seed everything
    pl.seed_everything(cfg["randomSeed"])
    
    # parse attributes from local imports 
    cfg["modelTraining"]["loss_fn"] = getattr(T.nn.functional, cfg["modelTraining"]["loss_fn"])
    cfg["modelTraining"]["optimizer"] = getattr(T.optim, cfg["modelTraining"]["optimizer"])
    cfg["modelTraining"]["lr_scheduler"] = getattr(T.optim.lr_scheduler, cfg["modelTraining"]["lr_scheduler"])
    cfg["dataset"]["dataset_class"] = getattr(datasets, cfg["dataset"]["dataset_class"])
    cfg["dataset"]["image_pertubator"] = preprocessing.ImagePertubators[cfg["dataset"]["image_pertubator"]]
    
    print(cfg)
    
    # create optuna study
    study = optuna.create_study(direction="minimize")
    


if __name__ == "__main__":
    run_optuna()
    
    # EXPERIMENT_NAME = "bbox_backbone"
    # RUN_NAME = "version_1"
    # MODEL_CHECKPOINTS_PATH = "checkpoints"
    # RANDOM_SEED = 1
    
    # NUM_TRAIN_WORKERS = 8
    # NUM_VAL_WORKERS = 4
    # IMAGE_SHAPE = (224, 224) # Width x Height
    # BATCH_SIZE = 80
    # TRAIN_VAL_TEST_SPLIT = [0.8, 0.15, 0.05]
    
    
    # DATASET_ARGS = {
    # }
    # # dataset = COCO2017PanopticsDataset(
    # #     image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
    # #     panoptics_path="//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations",
    # #     output_image_shape_WH=IMAGE_SHAPE,
    # #     instance_images_output_shape_WH=(8, 8),
    # #     max_number_of_instances=OUTPUT_LENGTH,
    # #     load_segmentation_masks=False,
    # #     image_pertubator=ImagePertubators.BASIC(),
    # #     # limit_to_first_n = 320
    # # )
    # # dataset = WIDERFaceDataset(
    # #     output_image_shape_WH=IMAGE_SHAPE, 
    # #     max_number_of_faces=OUTPUT_LENGTH,
    # #     train_path="//192.168.2.206/data/datasets/WIDER-Face/train",
    # #     val_path="//192.168.2.206/data/datasets/WIDER-Face/val",
    # #     image_pertubator = ImagePertubators.BASIC(),
    # #     center_bbox = True
    # # )
    
    
    
    
    # # ---------------------------------------------------------------------------------------------------------------
    # # print("find lr ...")
    # # tuner = tuning.Tuner(trainer=trainer)
    # # lr_finder = tuner.lr_find(
    # #     model=model, 
    # #     train_dataloaders=train_dataloader, 
    # #     val_dataloaders=val_dataloader,
    # #     num_training=50
    # # )
    # # fig = lr_finder.plot(suggest=True)
    # # fig.show()
    # # input("Press Enter to continue ...")
    
    # # ---------------------------------------------------------------------------------------------------------------
    # print("fitting model ...")
    
    # trainer.fit(model=model, datamodule=data_module)
    