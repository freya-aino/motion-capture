import torch as T
import pytorch_lightning as pl
import lovely_tensors as lt
import optuna
import hydra
import copy

import motion_capture.data.datasets as datasets # WIDERFaceDataset, WFLWDataset, COFWColorDataset, MPIIDataset, COCO2017PersonKeypointsDataset, COCO2017PanopticsDataset, COCO2017WholeBodyDataset
import motion_capture.data.preprocessing as preprocessing

from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.tuner import tuning
from omegaconf import DictConfig, OmegaConf

from motion_capture.model.models import UpsampleCrossAttentionNetwork, find_best_checkpoint_path
from motion_capture.data.datamodules import BboxDataModule


# ---------------------------------------------------------------------------------------------------------------

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def objective(
    logger_cfg: dict,
    checkpoint_callback_cfg: dict,
    experiment_cfg: dict, 
    trial: optuna.Trial
    ):
    
    experiment_cfg = copy.deepcopy(experiment_cfg)
    
    print("load experiment config ...")
    experiment_cfg["modelTraining"]["optimizer_kwargs"]["lr"] = trial.suggest_float("lr", *experiment_cfg["modelTraining"]["optimizer_kwargs"]["lr"], log=True)
    experiment_cfg["modelTraining"]["optimizer_kwargs"]["weight_decay"] = trial.suggest_float("weight_decay", *experiment_cfg["modelTraining"]["optimizer_kwargs"]["weight_decay"], log=True)
    experiment_cfg["modelTraining"]["lr_scheduler_kwargs"]["T_max"] = trial.suggest_int("T_max", *experiment_cfg["modelTraining"]["lr_scheduler_kwargs"]["T_max"], log=True)
    experiment_cfg["modelTraining"]["lr_scheduler_kwargs"]["eta_min"] = trial.suggest_float("eta_min", *experiment_cfg["modelTraining"]["lr_scheduler_kwargs"]["eta_min"], log=True)
    
    experiment_cfg["stochasticWeightAveraging"]["swa_lrs"] = trial.suggest_float("swa_lrs", *experiment_cfg["stochasticWeightAveraging"]["swa_lrs"], log=True)
    experiment_cfg["stochasticWeightAveraging"]["annealing_epochs"] = trial.suggest_int("anealing_epochs", *experiment_cfg["stochasticWeightAveraging"]["annealing_epochs"], log=True)
    
    # ---------------------------------------------------------------------------------------------------------------
    
    print("initializing logger ...")
    logger = MLFlowLogger(**logger_cfg)
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing callbacks ...")
    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_cfg)
    swa = StochasticWeightAveraging(**experiment_cfg["stochasticWeightAveraging"])
    optuna_callback = OptunaPruning(trial=trial, monitor=checkpoint_callback_cfg["monitor"])
    # ---------------------------------------------------------------------------------------------------------------
    print("initializing trainer ...")
    trainer = pl.Trainer(
        **experiment_cfg["trainer"],
        logger = logger,
        callbacks = [checkpoint_callback, swa, optuna_callback],
    )
    # ---------------------------------------------------------------------------------------------------------------
    print("initialize model ...")
    if experiment_cfg["continueTraining"]:
        best_ckpt_pth = find_best_checkpoint_path(checkpoint_callback_cfg["dirpath"])
        model = UpsampleCrossAttentionNetwork.load_from_checkpoint(best_ckpt_pth)
    else:
        model = UpsampleCrossAttentionNetwork(**experiment_cfg["modelStructure"], **experiment_cfg["modelTraining"])
    # ---------------------------------------------------------------------------------------------------------------
    print("initialize dataset ...")
    data_module = BboxDataModule(**experiment_cfg["datamodule"])
    # ---------------------------------------------------------------------------------------------------------------
    print("fitting model ...")
    trainer.fit(model=model, datamodule=data_module)
    
    return trainer.callback_metrics[checkpoint_callback_cfg["monitor"]].item()


@hydra.main(config_path="configs/hydra", config_name="config", version_base=None)
def run_optuna(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    import pprint
    
    print(cfg.keys())
    pprint.PrettyPrinter(indent=2).pprint(cfg)
    
    if not cfg["experiment"]["experimentName"]:
        print("!! please select experiment")
        exit()
    
    logger_cfg = cfg["logger"]
    checkpointCallback_cfg = cfg["checkpointCallback"]
    experiment_cfg = cfg["experiment"]
    
    # seed everything
    pl.seed_everything(experiment_cfg["randomSeed"])
    
    # parse str attributes as python attributes
    experiment_cfg["modelTraining"]["loss_fn"] = getattr(T.nn.functional, experiment_cfg["modelTraining"]["loss_fn"])
    experiment_cfg["modelTraining"]["optimizer"] = getattr(T.optim, experiment_cfg["modelTraining"]["optimizer"])
    experiment_cfg["modelTraining"]["lr_scheduler"] = getattr(T.optim.lr_scheduler, experiment_cfg["modelTraining"]["lr_scheduler"])
    experiment_cfg["datamodule"]["dataset_class"] = getattr(datasets, experiment_cfg["datamodule"]["dataset_class"])
    experiment_cfg["datamodule"]["image_pertubator"] = preprocessing.ImagePertubators[experiment_cfg["datamodule"]["image_pertubator"]]
    
    
    # create optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        func = lambda trial: objective(logger_cfg=logger_cfg, checkpoint_callback_cfg=checkpointCallback_cfg, experiment_cfg=experiment_cfg, trial=trial),
        n_trials=10,
        
        )


if __name__ == "__main__":
    
    try:
        T.cuda.empty_cache()
        run_optuna()
    except Exception as e:
        print(e)
    finally:
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
    