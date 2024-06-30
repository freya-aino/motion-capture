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

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from motion_capture.data.datasets import WIDERFaceDataset, WFLWDataset, COFWColorDataset, MPIIDataset, COCO2017PersonKeypointsDataset, COCO2017PanopticsDataset, COCO2017WholeBodyDataset
from motion_capture.model.models import UpsampleCrossAttentionNetwork

# ---------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    '''
        Notes for training:
        - Backbone - trained to generalize information
        - Neck - training to sets of semantically similar tasks
        - Head - training to one singular narrow task
        
        backbone:
            - ImageNet
            - COCOPanoptics
        Neck + Head bboxes:
            - WIDEFace (bboxes)
            - WFLW (bboxes)
            - COCOWholeBody (bboxes)
            - MPII (bboxes)
        Neck + Head keypoints:
            - WFLW (keypoints)
            - COCOPersonKeypoints (keypoints)
            - MPII (keypoints)
        
    '''
    
    # ---------------------------------------------------------------------------------------------------------------
    # TMP PARAMS
    TRAINING_NAME = "backbone-general"
    VERSION = "version_1"
    CHECKPOINT_PATH = "checkpoints/"
    
    IMAGE_SHAPE = (448, 448) # Width x Height
    MAX_NUMBER_OF_INSTANCES = 12
    BATCH_SIZE = 32
    
    BACKONE_OUTPUT_SIZE = 512
    HEAD_LATENT_SIZE = 256
    NECK_OUTPUT_SIZE = 256
    
    NUM_WORKERS = 2
    
    # ---------------------------------------------------------------------------------------------------------------
    
    print("loading model ...")
    
    model = UpsampleCrossAttentionNetwork(
        output_size=4,
        output_length=MAX_NUMBER_OF_INSTANCES,
        backbone_output_size=BACKONE_OUTPUT_SIZE,
        neck_output_size=NECK_OUTPUT_SIZE,
        head_latent_size=HEAD_LATENT_SIZE,
        loss_fn=T.nn.functional.mse_loss
    )
    
    print("loading dataset ...")
    
    dataset = COCO2017PanopticsDataset(
        image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
        panoptics_path="//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations",
        output_image_shape_WH=IMAGE_SHAPE,
        instance_images_output_shape_WH=(112, 112),
        max_number_of_instances=MAX_NUMBER_OF_INSTANCES,
        load_segmentation_masks=False,
        limit_to_first_n = 320
    )
    train_dataset, val_dataset = data.random_split(dataset, [0.8, 0.2])
    
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn_bbox,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.collate_fn_bbox,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    
    
    print(f"training model on {len(dataset)} datapoints ...")
    
    # logger = TensorBoardLogger(
    #     save_dir="logs/",
    #     name=TRAINING_NAME, 
    #     version = VERSION
    # )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=2,
        monitor="val_loss",
        # filename="epoch{epoch}-{step}-{val_loss:.4f}",
        # verbose=True,
        # mode="min",
        # every_n_epochs=1,
    )
    
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0.0,
    #     patience=5,
    #     verbose=True,
    #     mode="min",
    #     stopping_threshold=1 / 100 * 0.2 # 0.2 % of the normalized image size
    # )
    
    trainer = pl.Trainer(
        accelerator = "gpu",
        # logger=logger,
        # logger = False,
        callbacks = [checkpoint_callback],
        max_epochs = 5,
        # fast_dev_run=True,
        # check_val_every_n_epoch=1,
        # num_sanity_val_steps=0,
        # log_every_n_steps=1,
        # gradient_clip_algorithm="norm",
        # default_root_dir=CHECKPOINT_PATH,
        # enable_checkpointing=True
        # gradient_clip_val=1.0,
    )
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    print(trainer.logged_metrics)
    print(checkpoint_callback.best_k_models)
    
    # import tqdm
    
    # for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
    #     try:
    #         x, y = batch
    #         assert x.shape == (BATCH_SIZE, 3, *IMAGE_SHAPE)
    #         assert y.shape == (BATCH_SIZE, MAX_NUMBER_OF_INSTANCES, 4)
            
    #     except Exception as e:
    #         traceback.print_exc()
    #     pass
    
    
    # model = model.to("cuda")
    # for batch_i, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="training"):
        
    #     cu_batch = (batch[0].to("cuda"), batch[1].to("cuda"))
        
    #     model.training_step(cu_batch, batch_idx=batch_i)
        
    
    
    # TODO: offload the experiment configuration to some 3. party tooling
    # config = json.load(open("./model/config.json", "r+"))
    # SPATIAL_ENCODER_CONFIG = config["model"]["spatial_encoder"]
    # TRAINING_CONFIG = config["training"]
    # DEVICE = config["device"]
    
    # # MODEL
    # spatial_encoder_backbone = torch_models.resnet18(weights = torch_models.ResNet18_Weights.DEFAULT)
    # spatial_encoder = SpatialEncoder(backbone=spatial_encoder_backbone, config=SPATIAL_ENCODER_CONFIG)
    # spatial_encoder = spatial_encoder.eval().to(DEVICE)
    # # DATASET & SAMPLER & DATALOADER
    # wider_face_dataset = WIDERFaceDataset(
    #     image_shape=SPATIAL_ENCODER_CONFIG["image_encoder"]["input_shape"], 
    #     max_number_of_faces=SPATIAL_ENCODER_CONFIG["spatial_transformer"]["output_size"] // 4)
    
    # wider_face_indecies = [*range(len(wider_face_dataset))]
    # train_split = int(len(wider_face_dataset) * TRAINING_CONFIG["train_split"])
    # wider_face_sampler_train = data.SubsetRandomSampler(wider_face_indecies[:train_split])
    # wider_face_sampler_test = data.SubsetRandomSampler(wider_face_indecies[train_split:])

    # wider_face_dataloader_train = data.DataLoader(
    #     wider_face_dataset, 
    #     num_workers=16,
    #     pin_memory=True,
    #     sampler=wider_face_sampler_train, batch_size=TRAINING_CONFIG["batch_size"])
    # wider_face_dataloader_test = data.DataLoader(
    #     wider_face_dataset, 
    #     num_workers=16,
    #     pin_memory=True,
    #     sampler=wider_face_sampler_test, batch_size=TRAINING_CONFIG["batch_size"])


    # # OPTIMIZER & METRICS
    # spatial_encoder_opt = T.optim.RAdam(spatial_encoder.parameters())
    # prediction_loss_f = nn.SmoothL1Loss()
    
    # for epoch_i in range(TRAINING_CONFIG["epochs"]):

    #     print(f"epoch: {epoch_i}")

    #     spatial_encoder.train()
    #     train_loss = 0
    #     for datapoint in wider_face_dataloader_train:

    #         # dt = time.time()

    #         X = datapoint["image"].to(DEVICE).unsqueeze(1)
    #         Y = datapoint["face_bbox"].to(DEVICE).unsqueeze(1)

    #         y_hat = spatial_encoder(X)


    #         loss = prediction_loss_f(y_hat, Y)
    #         loss = loss.nanmean()

    #         train_loss += loss.detach().to("cpu").item()

    #         spatial_encoder_opt.zero_grad()
    #         loss.backward()
    #         spatial_encoder_opt.step()

    #         if "cuda" in DEVICE:
    #             T.cuda.synchronize()

    #         # print(f"time taken: {time.time() - dt}")

    #     print(f"\ttraining loss:  \t{train_loss / len(wider_face_dataloader_train)}")


    #     spatial_encoder.eval()
    #     test_loss = 0
    #     for datapoint in wider_face_dataloader_test:

    #         X = datapoint["image"].to(DEVICE).unsqueeze(1)
    #         Y = datapoint["face_bbox"].to(DEVICE).unsqueeze(1)

    #         y_hat = spatial_encoder(X)

    #         loss = prediction_loss_f(y_hat, Y)
    #         loss = loss.nanmean()

    #         test_loss += loss.detach().to('cpu').item()

    #         if "cuda" in DEVICE:
    #             T.cuda.synchronize()


    #     print(f"\tvalidation loss:\t{test_loss / len(wider_face_dataloader_test)}")


    #     if epoch_i != 0 and epoch_i % 10 == 0:
    #         print("saving model ...")
    #         T.save(spatial_encoder.state_dict(), os.path.join(MODEL_SAVE_PATH, f"{epoch_i}.pth"))