import torch as T
import pytorch_lightning as pl
import hydra

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf


from motion_capture.model.trainingmodules import BBoxTrainingModule
from motion_capture.core.utils import find_best_checkpoint_path
from motion_capture.data.datamodules import DataModule
from motion_capture.data.datasets import WIDERFace

# ---------------------------------------------------------------------------------------------------------------

@hydra.main(config_path="configs/hydra", config_name="face-detection", version_base=None)
def run(conf: DictConfig):
    conf = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(conf)
    
    assert conf.experiment.experimentName and conf.experiment.runName, "please select experiment and run name in the experiment config !"
    
    pl.seed_everything(conf.randomSeed)
    
    trainer = pl.Trainer(
        **conf.trainer,
        logger=MLFlowLogger(**conf.logger),
        callbacks=ModelCheckpoint(**conf.checkpointCallback)
    )
    
    # --------------- model setup ----------------
    print("initialize model ...")
    
    if conf.resumeTraining:
        print("resume training ...")
        model_path = find_best_checkpoint_path(conf.checkpoint_callback.dirpath, min_loss=True)
        model = BBoxTrainingModule.load_from_checkpoint(model_path).to(conf.trainer.accelerator)
    else:
        print("start new training ...")
        model = BBoxTrainingModule(**conf.model, **conf.training)
    
    # --------------- data setup ----------------
    print("initialize dataset ...")
    
    ## -------------- FACE BOUNDING BOX ----------------
    ## -------------- FACE INDICATORS ----------------
    
    # celeba_dataset = CelebA(
    #     annotatin_path="\\\\192.168.2.206\\data\\datasets\\CelebA\\Anno",
    #     image_path="\\\\192.168.2.206\\data\\datasets\\CelebA\\img\\img_align_celeba\\img_celeba",
    #     image_shape_WH = image_shape
    # )
    wider_face_dataset = WIDERFace(
        path="//192.168.2.206/data/datasets/WIDER-Face",
        image_shape_WH=conf.inputImageShape,
        max_number_of_faces=conf.maxNumberOfFaces
    )
    
    ## -------------- PERSON BOUNDING BOX ----------------
    ## -------------- PERSON SEGMENTATION ----------------
    
    # person_instance_dataset = COCO2017GlobalPersonInstanceSegmentation(
    #     image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
    #     annotation_folder_path = "//192.168.2.206/data/datasets/COCO2017/annotations",
    #     image_shape_WH=(224, 224),
    #     max_num_persons=1,
    #     max_segmentation_points=200
    # )
    
    ## -------------- CROP PERSON FACE & HANDS ----------------
    
    # coco_wholebody_dataset = COCO2017PersonWholeBody(
    #     annotations_folder_path="//192.168.2.206/data/datasets/COCO2017/annotations",
    #     image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
    #     image_shape_WH=image_shape,
    #     min_person_bbox_size=100
    # )
    
    data_module = DataModule(
        dataset=wider_face_dataset,
        y_key="bboxes", 
        **conf.datamodule
    )
    data_module.setup("fit")
    
    print(f"train on {len(wider_face_dataset)} samples")
    
    # --------------- training ----------------
    
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
    
    
    # from ghostface 
    # - ArcFace
    # - SGD(0.1, 0.9) with CosineAnnealingLR(50, 1e-5)
    # - 50 epochs
    # - l2 regularization to the output layer
    # - cosin distance used for verification
    # - mixed precision training
    
    
    # two step training !
    # 1. just train the head until converged
    # 2. finetue the entire network including the backbone
    
    
    # from RePFormer
    # - multiple predictions corresonding to using different output layers of the backbone (+ upsampled information)
    # - using the stages of the backbone to predict reesiduals to the previous stage
    
    

if __name__ == "__main__":
    
    run()
    T.cuda.empty_cache()
    