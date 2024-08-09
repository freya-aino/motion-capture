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

@hydra.main(config_path="configs/hydra", config_name="config", version_base=None)
def run(conf: DictConfig):
    conf = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(conf)
    
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
    from motion_capture.data.datasets import WIDERFace
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
    #     image_shape_WH=(224, 224),
    #     max_number_of_faces=conf.model.head.output_sequence_length
    # )
    # wflw_dataset = WFLW(
    #     image_shape_WH=image_shape, 
    #     path="//192.168.2.206/data/datasets/WFLW",
    #     max_number_of_faces=10
    # )
    
    ## -------------- PERSON BOUNDING BOX ----------------
    ## -------------- PERSON SEGMENTATION ----------------
    
    person_instance_dataset = COCO2017GlobalPersonInstanceSegmentation(
        image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
        annotation_folder_path = "//192.168.2.206/data/datasets/COCO2017/annotations",
        image_shape_WH=(224, 224),
        max_num_persons=1,
        max_segmentation_points=200
    )
    
    ## -------------- CROP PERSON FACE & HANDS ----------------
    
    # coco_wholebody_dataset = COCO2017PersonWholeBody(
    #     annotations_folder_path="//192.168.2.206/data/datasets/COCO2017/annotations",
    #     image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
    #     image_shape_WH=image_shape,
    #     min_person_bbox_size=100
    # )
    
    dataloader = DataModule(
        dataset=person_instance_dataset,
        y_key="bboxes",
        **conf.training.datamodule
    )
    
    print(f"train on {len(person_instance_dataset)} samples")
    
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    
    run()
    T.cuda.empty_cache()
    
    