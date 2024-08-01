import os
import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import hydra

import torch.utils.data as Tdata
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from motion_capture.data.preprocessing import ImageAugmentations
from motion_capture.model.models import VisionModel
from motion_capture.data.datasets import COCO2017GlobalPersonInstanceSegmentation

# ---------------------------------------------------------------------------------------------------------------

def find_best_checkpoint_path(checkpoint_dir, min_loss: bool = True, pattern="*.ckpt"):
    from glob import glob
    import re
    
    files = glob(os.path.join(checkpoint_dir, pattern))
    
    if len(files) == 0:
        return None
    
    all_models = []
    for file in files:
        ckpt = T.load(file, map_location=T.device("cpu"))
        for key, val in ckpt.get("callbacks", {}).items():
            if key.startswith("ModelCheckpoint"):
                all_models.append({
                    "model_path": val["best_model_path"],
                    "model_score": val["best_model_score"]
                })
    if min_loss:
        best_model = min(all_models, key=lambda x: x["model_score"])
    else:
        best_model = max(all_models, key=lambda x: x["model_score"])
    
    print(f"found best model with loss: {best_model['model_score']} from {best_model['model_path']}")
    return best_model["model_path"]

class DataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        dataset: Tdata.Dataset,
        image_augmentation: str,
        batch_size: int,
        train_val_test_split: tuple[float, float, float],
        num_train_workers: int,
        num_val_workers: int):
        
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_train_workers = num_train_workers
        self.num_val_workers = num_val_workers
        
        self.image_augmentation = ImageAugmentations[image_augmentation]
        
    def setup(self):
        splits = Tdata.random_split(
            dataset = self.dataset, 
            lengths = self.train_val_test_split,
        )
        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        self.test_dataset = splits[2]
        
    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return T.stack(batch)
        
    def train_dataloader(self):
        return Tdata.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_train_workers,
            persistent_workers=True if self.num_train_workers > 0 else False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return Tdata.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_val_workers,
            persistent_workers=True if self.num_val_workers > 0 else False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return Tdata.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False,
        )

# ---------------------------------------------------------------------------------------------------------------

@hydra.main(config_path="configs/hydra", config_name="config", version_base=None)
def run(conf: DictConfig):
    
    assert conf.experiment.experimentName and conf.experiment.runName, "please select experiment and run name in the experiment config !"
    
    # --------------- trianing setup ----------------
    print("initialize training ...")
    pl.seed_everything(conf.experiment.randomSeed)
    
    logger = MLFlowLogger(**conf.training.logger)
    checkpoint_callback = ModelCheckpoint(**conf.training.checkpoint_callback)
    trainer = pl.Trainer(**conf.training.trainer, logger = logger, callbacks = [checkpoint_callback])
    
    # --------------- model setup ----------------
    print("initialize model ...")
    #     if experiment.continue_training:
#         best_ckpt_pth = models.find_best_checkpoint_path(experiment.checkpoint_callback.dirpath)
#         if best_ckpt_pth:
#             model = model_class.load_from_checkpoint(best_ckpt_pth)
#         else:
#             model = model_class(**model_cfg, **model_training_cfg)
#     else:
#         model = model_class(**model_cfg, **model_training_cfg)
    
    model = VisionModel(**conf.model)
    model.backbone = model.backbone.eval()
    
    # --------------- data setup ----------------
    print("initialize dataset ...")
    
    # person_instance_dataset = COCO2017GlobalPersonInstanceSegmentation(
    #     image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
    #     annotation_folder_path = "//192.168.2.206/data/datasets/COCO2017/annotations",
    #     image_shape_WH=(224, 224),
    #     max_num_persons=10,
    #     max_segmentation_points=100
    # )
    
    # datamodule = DataModule(
    #     dataset = person_instance_dataset,
    #     **conf.training.data_module
    # )
    
    
    
    # VQVAE DATA
    from motion_capture.data.datasets import COCO2017PersonKeypointsDataset, COCOPanopticsObjectDetection, HAKELarge
    
    coco_dataset = COCOPanopticsObjectDetection(
        image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
        panoptics_path = "//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations",
        image_shape_WH=image_shape,
        max_number_of_instances=100
    ) # 120k images
    
    person_keypoints_dataset = COCO2017PersonKeypointsDataset(
        image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
        annotation_folder_path = "//192.168.2.206/data/datasets/COCO2017/annotations",
        image_shape_WH = image_shape,
        min_person_bbox_size = 100
    ) # 70k images
    
    hake_dataset = HAKELarge(
        annotation_path = "\\\\192.168.2.206\\data\\datasets\\HAKE\\Annotations",
        image_path = "\\\\192.168.2.206\\data\\datasets\\HAKE-large",
        image_shape_WH = image_shape,
    ) # 100k images
    
    # "\\192.168.2.206\data\datasets\CelebA\img\img_align_celeba\img_celeba" # 200k images
    
    
    
    
    
# @hydra.main(config_path="configs/hydra", config_name="config", version_base=None)
# def run(cfg: DictConfig):
#     # ---------------------------------------------------------------------------------------------------------------
#     print("initialize dataset ...")
#     datamodule_cfg["image_augmentation"] = preprocessing.ImageAugmentations[datamodule_cfg["image_augmentation"]]
#     selected_datasets = []
#     for dataset_k in data_cfg:
#         dataset_class = getattr(datasets, data_cfg[dataset_k]["class"])
#         selected_datasets.append(dataset_class(**data_cfg[dataset_k]["kwargs"]))
#     daatset = datasets.CombinedDataset(selected_datasets)
#     data_module = data_module_class(dataset = daatset, **datamodule_cfg)
#     data_module.setup()
#     # ---------------------------------------------------------------------------------------------------------------
#     print("fitting model ...")
#     trainer.fit(model=model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
#     # ---------------------------------------------------------------------------------------------------------------
#     print("testing model ...")
#     trainer.test(model=model, dataloaders=data_module.test_dataloader())
    
#     return trainer.callback_metrics["test_loss"].item()

if __name__ == "__main__":
    
    run()
    T.cuda.empty_cache()
    
    