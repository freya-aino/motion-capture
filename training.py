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
    IMAGE_SHAPE = (224, 224) # Width x Height
    MAX_NUMBER_OF_INSTANCES = 12
    BATCH_SIZE = 16
    
    BACKONE_OUTPUT_SIZE = 512
    HEAD_LATENT_SIZE = 128
    NECK_OUTPUT_SIZE = 128
    
    # ---------------------------------------------------------------------------------------------------------------
    
    print("loading model ...")
    
    model = UpsampleCrossAttentionNetwork(
        output_size=4,
        output_length=MAX_NUMBER_OF_INSTANCES,
        backbone_output_size=BACKONE_OUTPUT_SIZE,
        neck_output_size=NECK_OUTPUT_SIZE,
        head_latent_size=HEAD_LATENT_SIZE,
        loss_fn=T.nn.functional.l1_loss
    )
    
    print("loading dataset ...")
    
    dataset = COCO2017PanopticsDataset(
        image_folder_path="//192.168.2.206/data/datasets/COCO2017/images",
        panoptics_path="//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations",
        output_image_shape_WH=IMAGE_SHAPE,
        instance_images_output_shape_WH=(112, 112),
        max_number_of_instances=MAX_NUMBER_OF_INSTANCES,
        load_val_only=True
    )
    train_dataset, val_dataset = data.random_split(dataset, [0.8, 0.2])
    
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn_bbox,
        num_workers=8,
        persistent_workers=True
    )
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.collate_fn_bbox,
        # num_workers=8,
        # persistent_workers=True
    )
    
    
    print("training model ...")
    
    trainer = pl.Trainer(accelerator="gpu", max_epochs=10)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
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