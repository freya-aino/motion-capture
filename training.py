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
import torchvision.models as torch_models

from model.endtoendmodels import SpatialEncoder
from data.datasets import WIDERFaceDataset


# ---------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    '''
        Notes for training:
        - Backbone - should be trained to generalize image information
        - Neck - should be trained for face detection, keypoint detection and any other semantically similar tasks
        - Head - should be only trained on one singular narrow task
    '''


    # TODO: offload the experiment configuration to some 3. party tooling
    # TODO: use pytorch lighting for the configuration and store it seperately

    # MODEL_NAME = "spatial_encoder_henry"
    # EXPERIMENT_ID = "testing_trainability"

    # MODEL_SAVE_PATH = f".\\model_saves\\{MODEL_NAME}\\{EXPERIMENT_ID}"
    # if not os.path.exists(MODEL_SAVE_PATH):
    #     print(f"creating: {MODEL_SAVE_PATH}")
    #     os.mkdir(MODEL_SAVE_PATH)


    # config = json.load(open("./model/config.json", "r+"))

    # SPATIAL_ENCODER_CONFIG = config["model"]["spatial_encoder"]
    # TRAINING_CONFIG = config["training"]
    # DEVICE = config["device"]
    
    # MODEL
    spatial_encoder_backbone = torch_models.resnet18(weights = torch_models.ResNet18_Weights.DEFAULT)
    spatial_encoder = SpatialEncoder(backbone=spatial_encoder_backbone, config=SPATIAL_ENCODER_CONFIG)
    spatial_encoder = spatial_encoder.eval().to(DEVICE)

    # DATASET & SAMPLER & DATALOADER
    wider_face_dataset = WIDERFaceDataset(
        image_shape=SPATIAL_ENCODER_CONFIG["image_encoder"]["input_shape"], 
        max_number_of_faces=SPATIAL_ENCODER_CONFIG["spatial_transformer"]["output_size"] // 4)
    
    wider_face_indecies = [*range(len(wider_face_dataset))]
    train_split = int(len(wider_face_dataset) * TRAINING_CONFIG["train_split"])
    wider_face_sampler_train = data.SubsetRandomSampler(wider_face_indecies[:train_split])
    wider_face_sampler_test = data.SubsetRandomSampler(wider_face_indecies[train_split:])

    wider_face_dataloader_train = data.DataLoader(
        wider_face_dataset, 
        num_workers=16,
        pin_memory=True,
        sampler=wider_face_sampler_train, batch_size=TRAINING_CONFIG["batch_size"])
    wider_face_dataloader_test = data.DataLoader(
        wider_face_dataset, 
        num_workers=16,
        pin_memory=True,
        sampler=wider_face_sampler_test, batch_size=TRAINING_CONFIG["batch_size"])


    # OPTIMIZER & METRICS
    spatial_encoder_opt = T.optim.RAdam(spatial_encoder.parameters())
    prediction_loss_f = nn.SmoothL1Loss()


    # TODO:
    # - after writign this, seperate it into its own component
    
    for epoch_i in range(TRAINING_CONFIG["epochs"]):

        print(f"epoch: {epoch_i}")

        spatial_encoder.train()
        train_loss = 0
        for datapoint in wider_face_dataloader_train:

            # dt = time.time()

            X = datapoint["image"].to(DEVICE).unsqueeze(1)
            Y = datapoint["face_bbox"].to(DEVICE).unsqueeze(1)

            y_hat = spatial_encoder(X)


            loss = prediction_loss_f(y_hat, Y)
            loss = loss.nanmean()

            train_loss += loss.detach().to("cpu").item()

            spatial_encoder_opt.zero_grad()
            loss.backward()
            spatial_encoder_opt.step()

            if "cuda" in DEVICE:
                T.cuda.synchronize()

            # print(f"time taken: {time.time() - dt}")

        print(f"\ttraining loss:  \t{train_loss / len(wider_face_dataloader_train)}")


        spatial_encoder.eval()
        test_loss = 0
        for datapoint in wider_face_dataloader_test:

            X = datapoint["image"].to(DEVICE).unsqueeze(1)
            Y = datapoint["face_bbox"].to(DEVICE).unsqueeze(1)

            y_hat = spatial_encoder(X)

            loss = prediction_loss_f(y_hat, Y)
            loss = loss.nanmean()

            test_loss += loss.detach().to('cpu').item()

            if "cuda" in DEVICE:
                T.cuda.synchronize()


        print(f"\tvalidation loss:\t{test_loss / len(wider_face_dataloader_test)}")


        if epoch_i != 0 and epoch_i % 10 == 0:
            print("saving model ...")
            T.save(spatial_encoder.state_dict(), os.path.join(MODEL_SAVE_PATH, f"{epoch_i}.pth"))