import os
import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import tqdm
import timm

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

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
    
    from motion_capture.model.modules import VisionModule
    
    model = VisionModule(**conf.model)
    model = model.train().to("cuda")
    
    # --------------- data setup ----------------
    print("initialize dataset ...")
    
    from motion_capture.data.datamodules import DataModule
    from motion_capture.data.datasets import COCO2017GlobalPersonInstanceSegmentation
    
    ## -------------- FACE BOUNDING BOX ----------------
    ## -------------- FACE INDICATORS ----------------
    
    # celeba_dataset = CelebA(
    #     annotatin_path="\\\\192.168.2.206\\data\\datasets\\CelebA\\Anno",
    #     image_path="\\\\192.168.2.206\\data\\datasets\\CelebA\\img\\img_align_celeba\\img_celeba",
    #     image_shape_WH = image_shape
    # )
    # wider_face_dataset = WIDERFace(
    #     path="//192.168.2.206/data/datasets/WIDER-Face",
    #     image_shape_WH=image_shape,
    #     max_number_of_faces=10
    # )
    # wflw_dataset = WFLW(
    #     image_shape_WH=image_shape, 
    #     path="//192.168.2.206/data/datasets/WFLW",
    #     max_number_of_faces=10
    # )
    
    ## -------------- PERSON BOUNDING BOX ----------------
    ## -------------- PERSON SEGMENTATION ----------------
    
    # person_instance_dataset = COCO2017GlobalPersonInstanceSegmentation(
    #     image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
    #     annotation_folder_path = "//192.168.2.206/data/datasets/COCO2017/annotations",
    #     image_shape_WH=image_shape,
    #     max_num_persons=10,
    #     max_segmentation_points=100
    # )
    
    ## -------------- CROP PERSON KEYPOINTS ----------------
    
    # coco_wholebody_dataset = COCO2017PersonWholeBodyDataset(
    #     annotations_folder_path="//192.168.2.206/data/datasets/COCO2017/annotations",
    #     image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
    #     image_shape_WH=image_shape
    # )
    
    
    
    
    DataModule(
        dataset = 
        **conf.training.data_module
    )
    
    
    
    
    
    # print(f"dataset size: {len(dataloader)}")
    # trainer.fit(model, dataloader)
    

if __name__ == "__main__":
    
    run()
    T.cuda.empty_cache()
    
    