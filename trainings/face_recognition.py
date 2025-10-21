from tkinter import CHECKBUTTON
import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as Tdata

from motion_capture.utils.utils import load_timm_model
from motion_capture.data.preprocessing import ImageAugmentations
from motion_capture.utils.WiseIoU import IouLoss
from motion_capture.utils.sam import SAM
from motion_capture.model import PyramidTransformerHead
from motion_capture.utils.utils import find_best_checkpoint_path
from motion_capture.data.datasets import WIDERFace

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# ---------------------------------------------------------------------------------------------------------------


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        y_key,
        batch_size,
        image_augmentation,
        train_val_split,
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.y_key = y_key
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.image_augmentation = ImageAugmentations.get(
            image_augmentation, ImageAugmentations["NONE"]
        )

    def collate_fn(self, batch):
        x = T.stack([self.image_augmentation(b[0]) for b in batch])
        y = T.stack([b[1][self.y_key] for b in batch])
        return x, y

    def setup(self, stage):
        splits = Tdata.random_split(
            dataset=self.dataset,
            lengths=self.train_val_split,
        )
        self.train_dataset = splits[0]
        self.val_dataset = splits[1]

    def train_dataloader(self):
        return Tdata.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers != 1 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return Tdata.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers != 1 else False,
            pin_memory=True,
        )


class BBoxTrainingModule(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        head_kwargs: dict,
        iou_loss_type="IoU",
        finetune=False,
        optimizer_kwargs: dict = {},
        lr_scheduler_kwargs: dict = {},
    ):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.backbone = load_timm_model(
            backbone_name, pretrained=True, features_only=True
        )
        self.head = PyramidTransformerHead(**head_kwargs)
        self.iouloss = IouLoss(ltype=iou_loss_type, monotonous=None)
        self.finetune = finetune

        if not finetune:
            self.backbone.eval()

    def forward(self, x):
        backbone_out = self.backbone(x)[-3:]
        return self.head(backbone_out)

    def compute_loss(self, y_, y):
        y = y.reshape(-1, 4)

        area = (y[..., 2:4] - y[..., 0:2]).prod(dim=-1)
        target_area_valid = ~T.isclose(area, T.tensor(0.0))

        loss, iou = 0, 0
        valids = 0
        for y_i in y_:
            y_i = y_i.reshape(-1, 4)

            area = (y_i[..., 2:4] - y_i[..., 0:2]).prod(dim=-1)

            predicted_area_valid = ~T.isclose(area, T.tensor(0.0))
            m = predicted_area_valid & target_area_valid

            if m.sum() == 0:
                continue
            valids += 1

            # print("y_i[m]: ", y_i[m].shape)
            # print("y[m]: ", y[m].shape)

            losses = self.head.compute_loss(
                y_i[m], y[m], loss_fn=self.iouloss, ret_iou=True
            )

            ldif, liou = [*zip(*losses)]
            ldif = T.stack(ldif)
            liou = T.stack(liou)

            # print("ldif: ", ldif.shape)
            # print("liou: ", liou.shape)

            loss += (ldif - 1).mean()
            iou += (1 - liou).mean()

        loss /= valids
        iou /= valids

        return loss, iou

        # return self.head.calculate_loss(y_[m], y[m], loss_fn=F.l1_loss)

    def on_train_start(self):
        if not self.finetune:
            self.backbone.eval()
        else:
            self.backbone.train()

        self.head.train()
        self.iouloss.train()

        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch

        # --- SAM ---
        def closure():
            loss, iou = self.compute_loss(self(x), y)
            if loss == 0:
                print("no valid area")
                return None
            loss.backward()
            return loss

        loss, iou = self.compute_loss(self(x), y)
        if loss == 0:
            print("no valid area")
            return None
        loss.backward()
        opt.step(closure)
        opt.zero_grad()

        self.log("train_loss", loss)
        self.log("train_IoU", iou)

        return {"train_loss": loss, "train_IoU": iou}

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.log("learning_rate", lr_scheduler.get_last_lr()[0])
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        loss, iou = self.compute_loss(self(x), y)
        if loss == 0:
            print("no valid area")
            return None

        self.log("val_loss", loss)
        self.log("val_IoU", iou)
        return {"val_loss": loss}

    def configure_optimizers(self):
        if self.finetune:
            optim = SAM(
                self.parameters(),
                base_optimizer=T.optim.SGD,
                **self.hparams.optimizer_kwargs,
            )
        else:
            optim = SAM(
                self.head.parameters(),
                base_optimizer=T.optim.SGD,
                **self.hparams.optimizer_kwargs,
            )

        lr_scheduler = T.optim.lr_scheduler.CosineAnnealingLR(
            optim, **self.hparams.lr_scheduler_kwargs
        )

        return [optim], [lr_scheduler]


if __name__ == "__main__":
    # -- experiment
    EXPERIMENT_NAME = ""
    RUN_NAME = ""
    RANDOM_SEED = 0

    # -- general
    INPUT_IMAGE_SHAPE = [224, 224]
    MAX_NUMBER_OF_FACES = 5
    # resumeTraining: false

    # -- trainer
    ACCELERATOR = "cuda"
    MAX_EPOCHS = 50
    # precision: "16-mixed"

    # -- checkpointing
    CHECKPOINT_DIR = f"checkpoints/{EXPERIMENT_NAME}/{RUN_NAME}"

    # -- model
    BACKBONE_NAME = "convnextv2_atto.fcmae_ft_in1k"
    HEAD_INPUT_DIMS = 320
    HEAD_INPUT_LENGTH = 49
    HEAD_OUTPUT_DIMS = 4
    HEAD_OUTPUT_LENGTH = 5
    HEAD_NUM_HEADS = 4

    # -- training
    IOU_LOSS_TYPE = "EIoU"
    OPTIMIZER_ADAPTIVE = True
    OPTIMIZER_RHO = 0.4
    OPTIMIZER_LR = 0.1
    OPTIMIZER_MOMENTUM = 0.9
    OPTIMIZER_WEIGHT_DECAY = 0.0005
    LR_SCHEDULER_T_MAX = 50
    LR_SCHEDULER_ETA_MIN = 1e-5

    # -- datamodule
    IMAGE_AUGMENTATION = "INPLACE"
    BATCH_SIZE = 64
    TRAIN_VAL_SPLIT = [0.8, 0.2]
    NUM_WORKERS = 2

    pl.seed_everything(RANDOM_SEED)

    mlflow_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        save_dir="mlflow_runs",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        save_top_k=1,
        every_n_epochs=1,
        monitor="val_loss",
        filename="{epoch}-{step}-{val_loss:.4f}",
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        max_epochs=MAX_EPOCHS,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
    )

    # --------------- model setup ----------------
    print("initialize model ...")

    if conf.resumeTraining:
        print("resume training ...")
        model_path = find_best_checkpoint_path(
            conf.checkpoint_callback.dirpath, min_loss=True
        )
        model = BBoxTrainingModule.load_from_checkpoint(model_path).to(
            conf.trainer.accelerator
        )
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
        max_number_of_faces=conf.maxNumberOfFaces,
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
        dataset=wider_face_dataset, y_key="bboxes", **conf.datamodule
    )
    data_module.setup("fit")

    print(f"train on {len(wider_face_dataset)} samples")

    # --------------- training ----------------

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

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

    T.cuda.empty_cache()
