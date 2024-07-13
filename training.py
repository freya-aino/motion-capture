import lightning.pytorch
import torch as T
import pytorch_lightning as pl
import lovely_tensors as lt
import optuna
import hydra
import copy

import motion_capture.data.datasets as datasets
import motion_capture.model.models as models
import motion_capture.data.preprocessing as preprocessing
import motion_capture.data.datamodules as datamodules

from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.tuner import tuning
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------------------------------------------

# class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


@hydra.main(config_path="configs/hydra", config_name="config", version_base=None)
def run(cfg: DictConfig):
    
    import pprint
    pprint.PrettyPrinter(indent=2).pprint(OmegaConf.to_container(cfg, resolve=True))
    
    if not cfg.experiment.experimentName or not cfg.experiment.runName:
        print("!! please select experiment and run name !")
        exit()
    
    experiment = cfg.experiment
    
    # seed everything
    pl.seed_everything(experiment.randomSeed)
    
    # ---------------------------------------------------------------------------------------------------------------
    
    model_cfg = OmegaConf.to_container(experiment.model, resolve=True)
    model_training_cfg = OmegaConf.to_container(experiment.model_training, resolve=True)
    data_cfg = OmegaConf.to_container(experiment.data, resolve=True)
    datamodule_cfg = OmegaConf.to_container(experiment.datamodule, resolve=True)
    
    # parse str attributes as python attributes
    model_class = getattr(models, model_cfg["class"])
    model_training_cfg["loss_fn"] = getattr(T.nn.functional, model_training_cfg["loss_fn"])
    model_training_cfg["optimizer"] = getattr(T.optim, model_training_cfg["optimizer"])
    model_training_cfg["lr_scheduler"] = getattr(T.optim.lr_scheduler, model_training_cfg["lr_scheduler"])
    
    data_module_class = getattr(datamodules, datamodule_cfg["class"])
    
    # remove unessesary keys
    model_cfg.pop("class")
    datamodule_cfg.pop("class")
    
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing logger ...")
    logger = MLFlowLogger(**experiment.logger)
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing callbacks ...")
    checkpoint_callback = ModelCheckpoint(**experiment.checkpoint_callback)
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing trainer ...")
    trainer = pl.Trainer(**experiment.trainer, logger = logger, callbacks = [checkpoint_callback])
    # ---------------------------------------------------------------------------------------------------------------
    print("initialize model ...")
    if experiment.continue_training:
        best_ckpt_pth = models.find_best_checkpoint_path(experiment.checkpoint_callback.dirpath)
        if best_ckpt_pth:
            model = model_class.load_from_checkpoint(best_ckpt_pth)
        else:
            model = model_class(**model_cfg, **model_training_cfg)
    else:
        model = model_class(**model_cfg, **model_training_cfg)
    # ---------------------------------------------------------------------------------------------------------------
    print("initialize dataset ...")
    datamodule_cfg["image_augmentation"] = preprocessing.ImageAugmentations[datamodule_cfg["image_augmentation"]]
    selected_datasets = []
    for dataset_k in data_cfg:
        dataset_class = getattr(datasets, data_cfg[dataset_k]["class"])
        selected_datasets.append(dataset_class(**data_cfg[dataset_k]["kwargs"]))
    daatset = datasets.CombinedDataset(selected_datasets)
    data_module = data_module_class(dataset = daatset, **datamodule_cfg)
    data_module.setup()
    # ---------------------------------------------------------------------------------------------------------------
    print("fitting model ...")
    trainer.fit(model=model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
    # ---------------------------------------------------------------------------------------------------------------
    print("testing model ...")
    trainer.test(model=model, dataloaders=data_module.test_dataloader())
    
    return trainer.callback_metrics["test_loss"].item()

if __name__ == "__main__":
    
    run()
    T.cuda.empty_cache()
    
    
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
    