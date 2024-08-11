import random
import time
import math
import re
import os
import csv
import json
import cv2
import logging
import h5py

import numpy as np
import torch as T
import pandas as pd

from copy import deepcopy
from torch.utils import data
from torchvision.transforms.functional import resize, crop
from torchvision.io import read_image, ImageReadMode
from torch.nn.functional import one_hot, pad

# ------------------------------------------------------------------------

# Segmentation for classification: 
# - https://github.com/qubvel/segmentation_models.pytorch

# ------------------------------------------------------------------------

# ------------------------------------------------------------------------

def scale_points(points: T.Tensor, input_shape: tuple, output_shape: tuple) -> T.Tensor:
    '''
        points in format [N, (W, H)] or [N, (W, H, D)] (only W and H are relevant)
        input_shape and output_shape in format [W, H]
    '''
    assert len(points.shape) > 1, "points must have at least 2 dimensions"
    if type(input_shape) != T.Tensor:
        input_shape = T.tensor(input_shape, dtype=T.float32)
    if type(output_shape) != T.Tensor:
        output_shape = T.tensor(output_shape, dtype=T.float32)
    
    if points.shape[-1] == 2:
        return points / input_shape.expand(*points.shape[:-1], -1) * output_shape.expand(*points.shape[:-1], -1)
    else:
        return T.cat([points[..., :2] / input_shape.expand(*points.shape[:-1], -1) * output_shape.expand(*points.shape[:-1], -1), points[..., 2:]], dim=-1)

def center_bbox(bbox: T.Tensor) -> T.Tensor:
    '''
        bbox in format [x1, y1, w, h] or [[x, y], [w, h]]
    '''
    
    # bbox is in format [x1, y1, w, h]
    if bbox.dim() == 1:
        return T.tensor([bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2), bbox[2] / 2, bbox[3] / 2])
    
    # bbox is in format [[x, y], [w, h]]
    return T.tensor(
        [
            [bbox[0][0] + (bbox[1][0] / 2), bbox[0][1] + (bbox[1][1] / 2)], 
            [bbox[1][0] / 2, bbox[1][1] / 2]
        ]
    )

class CombinedDataset(data.Dataset):
    def __init__(self, datasets: list):
        super(type(self), self).__init__()
        self.datasets = datasets
    
    def __len__(self):
        return sum([len(d) for d in self.datasets])
    
    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)

# ------------------------------------------------------------------------

class CelebA(data.Dataset):
    def __init__(self, annotatin_path, image_path, image_shape_WH):
        super().__init__()
        self.keypoint_indecies = ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y", 
            "nose_x", "nose_y", "leftmouth_x", "leftmouth_y",
            "rightmouth_x", "rightmouth_y"]
        self.bbox_indecies = ["x_1", "y_1", "width", "height"]
        # self.attribute_indecies = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
        # "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
        # "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
        # "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        # "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
        # "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
        # "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
        # "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
        # "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
        # "Wearing_Necktie", "Young"]
        
        self.image_shape = image_shape_WH
        
        landmarks = pd.read_csv(os.path.join(annotatin_path, "list_landmarks_celeba.txt"), header=1, delimiter=r"\s+")
        # attributes = pd.read_csv(os.path.join(annotatin_path, "list_attr_celeba.txt"), header=1, delimiter=r"\s+")
        bboxes = pd.read_csv(os.path.join(annotatin_path, "list_bbox_celeba.txt"), header=1, delimiter=r"\s+")
        
        bboxes.index = bboxes["image_id"]
        # self.all_datapoints = attributes.join([landmarks, bboxes])
        self.all_datapoints = landmarks.join(bboxes)
        self.all_datapoints.reset_index(inplace=True)
        
        self.all_datapoints["imagePath"] = image_path + "/" + self.all_datapoints["image_id"]
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        datapoint = self.all_datapoints.iloc[idx]
        
        image = read_image(datapoint["imagePath"], ImageReadMode.RGB)
        keypoints = T.tensor(datapoint[self.keypoint_indecies].to_numpy().astype(np.float32)).reshape(-1, 2)
        keypoints = scale_points(keypoints, image.shape[::-1][:2], [1, 1])
        
        bbox = T.tensor(datapoint[self.bbox_indecies].to_numpy().astype(np.float32)).reshape(2, 2)
        bbox = scale_points(bbox, image.shape[::-1][:2], [1, 1])
        bbox[1, :] += bbox[0, :]
        bbox = bbox.flatten()
        
        image = resize(image, self.image_shape[::-1], antialias=True) / 255
        
        return image, {
            "keypoints": keypoints,
            "bbox": bbox
        }

class WIDERFace(data.Dataset):
    
    def __init__(self, path: str, image_shape_WH: tuple, max_number_of_faces: int):
        
        super(type(self), self).__init__()
        
        self.image_shape = image_shape_WH
        self.max_number_of_faces = max_number_of_faces
        
        train_image_path = os.path.join(path, "train", "images")
        val_image_path = os.path.join(path, "val", "images")
        
        train_annotations = pd.read_csv(os.path.join(path, "train", "train_bbx_gt.txt"), header=None)[0]
        val_annotations = pd.read_csv(os.path.join(path, "val", "val_bbx_gt.txt"), header=None)[0]
        
        annotations = pd.concat([train_annotations, val_annotations], ignore_index=True)
        is_file = annotations.map(lambda x: ".jpg" in x).to_numpy()
        is_train = annotations.index < len(train_annotations)
        
        # add image path
        train_file_mask = is_train & is_file
        val_file_mask = (~is_train) & is_file
        annotations[train_file_mask] = train_image_path + "/" + annotations[train_file_mask]
        annotations[val_file_mask] = val_image_path + "/" + annotations[val_file_mask]
        
        # format bbox infos
        is_num_bboxes = annotations.map(lambda x: x.isnumeric())
        bbox_info_mask = (~is_num_bboxes) & (~is_file)
        format_fn = lambda x: np.array(x.split(" ")[:-1], dtype=int)
        annotations[bbox_info_mask] = annotations[bbox_info_mask].map(format_fn)
        
        chunk_indecies = annotations[is_file].index.to_numpy()
        chunk_sizes = annotations[chunk_indecies + 1].values.astype(int)
        chunks = (
            annotations[chunk_i:chunk_i+chunk_size+2].to_list()
            for chunk_i, chunk_size in zip(chunk_indecies, chunk_sizes) 
            if chunk_size > 0 and chunk_size <= max_number_of_faces
        )
        
        self.all_datapoints = list(chunks)
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        
        datapoint = self.all_datapoints[idx]
        image_path = datapoint[0]
        num_faces = int(datapoint[1])
        faces = T.tensor(np.stack(datapoint[2:]), dtype=T.long)
        
        image = read_image(image_path, mode=ImageReadMode.RGB)
        bboxes = faces[:, 0:4].reshape(-1, 2, 2)
        blur = pad(one_hot(faces[:, 4], 3), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        typical_expression = pad(one_hot(faces[:, 5], 2), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        normal_illumination = pad(one_hot(faces[:, 6], 2), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        is_valid = pad(one_hot(1 - faces[:, 7], 2), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        occlusion = pad(one_hot(faces[:, 8], 3), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        typical_pose = pad(one_hot(faces[:, 9], 2), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        
        # scale bboxes
        bboxes = scale_points(bboxes, image.shape[::-1][:2], [1, 1])
        bboxes[:, 1, :] = (bboxes[:, 0, :] + bboxes[:, 1, :])
        bboxes = pad(bboxes.reshape(-1, 4), (0, 0, 0, self.max_number_of_faces - num_faces), value=0)
        
        # resize image
        image = resize(image, self.image_shape[::-1], antialias=True) / 255
        
        return image, {
            "bboxes": bboxes, 
            "blur": blur,
            "typicalExpression": typical_expression,
            "illumination": normal_illumination,
            "validity": is_valid,
            "occlusion": occlusion,
            "typicalPose": typical_pose
        }

class WFLW(data.Dataset):
    def __init__(self, image_shape_WH: tuple, path: str, max_number_of_faces: int):
        super(type(self), self).__init__()
        """
        pose:
            normal pose->0
            large pose->1
        expression:
            normal expression->0
            exaggerate expression->1
        illumination:
            normal illumination->0
            extreme illumination->1
        make-up:
            no make-up->0
            make-up->1
        occlusion:
            no occlusion->0
            occlusion->1
        blur:
            clear->0
            blur->1
        """
        
        self.max_number_of_faces = max_number_of_faces
        self.image_shape = image_shape_WH
        image_folder_path = os.path.join(path, "images")
        
        train_datapoints = pd.read_csv(os.path.join(path, "annotations", "train.txt"), header=None)[0]
        validation_datapoints = pd.read_csv(os.path.join(path, "annotations", "validation.txt"), header=None)[0]
        
        datapoints = pd.concat([train_datapoints, validation_datapoints], ignore_index=True)
        datapoints = datapoints.map(lambda x: x.split(" "))
        
        self.all_datapoints = pd.DataFrame()
        self.all_datapoints["imagePath"] = datapoints.map(lambda x: os.path.join(image_folder_path, x[-1]))
        self.all_datapoints["bbox"] = datapoints.map(lambda x: np.array(x[196:200], dtype=np.int16).reshape(2, 2))
        self.all_datapoints["keypoints"] = datapoints.map(lambda x: np.array(x[:196], dtype=np.float32).reshape(-1, 2))
        self.all_datapoints["indicators"] = datapoints.map(lambda x: np.array(x[200:206], dtype=np.int16))
        
        self.all_datapoints = [
            (image_path, {k: list(v.values()) for k, v in elements.to_dict().items()})
            for image_path, elements in 
            self.all_datapoints.groupby("imagePath").filter(lambda x: len(x["bbox"]) <= self.max_number_of_faces).groupby("imagePath")
        ]
    
    def __len__(self):
        return (len(self.all_datapoints))
    
    def __getitem__(self, idx):
        image_path, datapoints = self.all_datapoints[idx]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        
        bboxes = np.stack(datapoints["bbox"])
        bboxes = scale_points(bboxes, image.shape[::-1][:2], [1, 1]).to(dtype=T.float32)
        bboxes = pad(bboxes[:self.max_number_of_faces].reshape(-1, 4), (0, 0, 0, max(0, self.max_number_of_faces - len(bboxes))), value=0)
        
        keypoints = T.tensor(np.stack(datapoints["keypoints"])).to(dtype=T.float32)
        keypoints = scale_points(keypoints, image.shape[::-1][:2], [1, 1])
        keypoints = pad(keypoints[:self.max_number_of_faces], (0, 0, 0, max(0, self.max_number_of_faces - len(keypoints))), value=0)
        
        indicators = T.tensor(np.stack(datapoints["indicators"]), dtype=T.long)
        
        pose = pad(one_hot(indicators[:, 0], 2), (0, 0, 0, max(0, self.max_number_of_faces - len(indicators))), value=0)
        expression = pad(one_hot(indicators[:, 1], 2), (0, 0, 0, max(0, self.max_number_of_faces - len(indicators))), value=0)
        illumination = pad(one_hot(indicators[:, 2], 2), (0, 0, 0, max(0, self.max_number_of_faces - len(indicators))), value=0)
        make_up = pad(one_hot(indicators[:, 3], 2), (0, 0, 0, max(0, self.max_number_of_faces - len(indicators))), value=0)
        occlusion = pad(one_hot(indicators[:, 4], 2), (0, 0, 0, max(0, self.max_number_of_faces - len(indicators))), value=0)
        blur = pad(one_hot(indicators[:, 5], 2), (0, 0, 0, max(0, self.max_number_of_faces - len(indicators))), value=0)
        
        validity = T.zeros(self.max_number_of_faces, dtype=T.long)
        validity[:len(keypoints)] = 1
        validity = one_hot(validity, 2)
        
        image = resize(image, self.image_shape[::-1], antialias=True) / 255
        
        return image, {
            "validity": validity,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "pose": pose,
            "expression": expression,
            "illumination": illumination,
            "makeUp": make_up,
            "occlusion": occlusion,
            "blur": blur
        }

# class COFWFaceDetection(data.Dataset):
#     def __init__(self, path: str, image_shape_WH: tuple):
#         super(type(self), self).__init__()
        
#         self.image_shape = image_shape_WH
        
#         train_datapoints = self.extract_formatted_datapoints(os.path.join(path, "color_train.mat"))
#         val_datapoints = self.extract_formatted_datapoints(os.path.join(path, "color_test.mat"))
        
#         self.all_datapoints = [*train_datapoints, *val_datapoints]
    
#     def extract_formatted_datapoints(self, path: str):
#         file = h5py.File(path, "r")
#         keys = list(file.get("/"))
        
#         image_refs = np.array(file.get(keys[1])).squeeze()
#         bboxes = T.tensor(np.array(file.get(keys[2])), dtype=T.int16).squeeze().T.reshape(-1, 2, 2)
#         images = (np.array(file[img_ref]) for img_ref in image_refs)
#         return zip(images, bboxes)
        
#     def __len__(self):
#         return len(self.all_datapoints)
    
#     def __getitem__(self, idx):
        
#         datapoint = self.all_datapoints[idx]
        
#         # load image
#         image = T.tensor(datapoint[0], dtype=T.uint8)
        
#         if len(image.shape) == 2:
#             return None
        
#         image = image.permute(0, 2, 1)
        
#         # scale keypoints and bbox acording to the new full image size
#         bbox = scale_points(datapoint[1], image.shape[::-1][:2], [1, 1])
#         bbox[1, :] = bbox[0, :] + bbox[1, :]
#         bbox = bbox.reshape(4)
        
#         # scale full image and face image
#         image = resize(image, self.image_shape[::-1], antialias=True) / 255
        
#         return image, {
#             "bbox": bbox
#         }


# class COFWColorDataset(data.Dataset):
#     def __init__(
#         self, 
#         output_full_image_shape_WH: tuple, 
#         output_face_image_shape_WH: tuple,
#         data_path: str,
#         center_bbox: bool = True):
        
#         super(type(self), self).__init__()
        
#         self.center_bbox = center_bbox
        
#         self.output_full_image_shape = output_full_image_shape_WH
#         self.output_face_image_shape = output_face_image_shape_WH
        
#         self.train_file, train_datapoints = self.extract_formatted_datapoints(os.path.join(data_path, "color_train.mat"), is_train=True)
#         self.val_file, val_datapoints = self.extract_formatted_datapoints(os.path.join(data_path, "color_test.mat"), is_train=False)
#         self.all_datapoints = [*train_datapoints, *val_datapoints]
        
#     def __len__(self):
#         return len(self.all_datapoints)
    
#     def __getitem__(self, idx):
        
#         datapoint = self.all_datapoints[idx]
        
#         bbox = datapoint["bbox"]
#         keypoints = datapoint["keypoints"]
#         visibility = datapoint["visibility"]
        
#         # load image, keypoints, visibility and bbox
#         image = self.train_file[datapoint["image_ref"]] if datapoint["is_train"] else self.val_file[datapoint["image_ref"]]
#         image = T.tensor(image)
#         image = image.permute(0, 2, 1)
        
#         # scale keypoints and bbox acording to the new full image size
#         full_scale_keypoints = scale_points(keypoints, image.shape[::-1][:2], self.output_full_image_shape)
#         full_scale_bbox = scale_points(bbox, image.shape[::-1][:2], self.output_full_image_shape)
        
#         if self.center_bbox:
#             full_scale_bbox = center_bbox(full_scale_bbox)
        
#         # crop face and scale keypoints
#         face_image = crop(image, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
#         local_scaled_keypoints = scale_points(keypoints - bbox[0], face_image.shape[::-1][:2], self.output_face_image_shape)
        
#         # scale full image and face image
#         face_image = resize(face_image, self.output_face_image_shape[::-1], antialias=True)
#         image = resize(image, self.output_full_image_shape[::-1], antialias=True)
        
#         return {
#             "fullImage": image,
#             "faceImage": face_image,
#             "faceBbox": full_scale_bbox,
#             "localKeypoints": local_scaled_keypoints,
#             "globalKeypoints": full_scale_keypoints,
#             "keypointOcclusion": visibility
#         }
    
#     def extract_formatted_datapoints(self, path: str, is_train: bool):
        
#         file = h5py.File(path, "r")
#         keys = list(file.get("/"))
        
#         image_refs = np.array(file.get(keys[1])).squeeze()
#         bboxes = T.tensor(np.array(file.get(keys[2])), dtype=T.int16).squeeze().T.reshape(-1, 2, 2)
#         phis = T.tensor(np.array(file.get(keys[3]))).squeeze().T
        
#         keypoints = phis[:, :58].reshape(-1, 2, 29).permute(0, 2, 1)
#         visible = (1 - phis[:, 58:]).to(dtype=T.bool)
        
#         return file, [({
#             "is_train": is_train,
#             "image_ref": p[0],
#             "bbox": p[1],
#             "keypoints": p[2],
#             "visibility": p[3]
#             }) for p in zip(image_refs, bboxes, keypoints, visible)]

class MPIIDataset(data.Dataset):
    
    def __init__(
        self,
        output_full_image_shape_WH: tuple,
        output_person_image_shape_WH: tuple,
        annotation_path: str,
        image_folder_path: str):
        super(type(self), self).__init__()
        
        self.output_full_image_shape = output_full_image_shape_WH
        self.output_person_image_shape = output_person_image_shape_WH
        
        self.center_bbox = center_bbox
        
        self.image_folder_path = image_folder_path
        
        with open(os.path.join(annotation_path, "trainval.json")) as f:
            self.datapoints = json.load(f)
        
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        dp = self.datapoints[idx]
        
        image = read_image(os.path.join(self.image_folder_path, dp["image"]), mode=ImageReadMode.RGB).to(dtype=T.float32)
        keypoints = T.tensor(dp["joints"])
        visibility = T.tensor(dp["joints_vis"], dtype=T.bool)
        
        # calculate bounding box from visible keypoints
        visible_keypoints = keypoints[visibility == 1]
        bbox = T.stack([visible_keypoints.min(dim=0)[0], visible_keypoints.max(dim=0)[0]])
        bbox = T.cat([bbox[0], bbox[1] - bbox[0]]).to(dtype=T.int16).reshape(2, 2)
        
        # scale bbox and keypoints to full image size
        full_scaled_bbox = scale_points(bbox, image.shape[::-1][:2], self.output_full_image_shape)
        full_scaled_keypoints = scale_points(keypoints, image.shape[::-1][:2], self.output_full_image_shape)
        
        full_scaled_bbox[1, :] = full_scaled_bbox[0, :] + full_scaled_bbox[1, :]
        
        # crop image and scale image and keyponits to person image size
        person_image = crop(image, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
        local_scaled_keypoints = scale_points(keypoints - bbox[0], person_image.shape[::-1][:2], self.output_person_image_shape)
        
        # scale full image and person image
        person_image = resize(person_image, self.output_person_image_shape[::-1])
        image = resize(image, self.output_full_image_shape[::-1])
        
        return {
            "fullImage": image,
            "personImage": person_image,
            "personBbox": full_scaled_bbox,
            "globalKeypoints": full_scaled_keypoints,
            "localKeypoints": local_scaled_keypoints,
            "keypointVisibility": visibility,
        }

class COCO2017GlobalPersonInstanceSegmentation(data.Dataset):
    
    def __init__(
        self,
        image_folder_path: str,
        annotation_folder_path: str,
        image_shape_WH: tuple,
        max_num_persons: int,
        max_segmentation_points: int = 100,
        min_bbox_size: int = 50):
        
        super().__init__()
        
        self.max_num_persons = max_num_persons
        self.max_segmentation_points = max_segmentation_points
        self.image_shape = image_shape_WH
        
        with open(os.path.join(annotation_folder_path, "person_keypoints_train2017.json"), "r") as f:
            train_json = json.load(f)
        with open(os.path.join(annotation_folder_path, "person_keypoints_val2017.json"), "r") as f:
            val_json = json.load(f)
        
        images = pd.DataFrame.from_records((*train_json["images"], *val_json["images"]))
        annotations = pd.DataFrame.from_records((*train_json["annotations"], *val_json["annotations"]))
        
        self.all_datapoints = pd.merge(
            left = annotations,
            right = images,
            left_on = "image_id",
            right_on = "id",
        )
        
        self.all_datapoints["image_path"] = image_folder_path + "/" + self.all_datapoints["file_name"]
        self.all_datapoints = self.all_datapoints[self.all_datapoints["bbox"].apply(lambda x: x[2] * x[3] > (min_bbox_size**2))]
        self.all_datapoints = self.all_datapoints[self.all_datapoints["segmentation"].apply(type) == list]
        
        self.all_datapoints.drop(columns=[
            "num_keypoints", "area", "iscrowd", "keypoints",
            "image_id", "category_id", "id_x", "license", "file_name",
            "coco_url", "height", "width", "date_captured", "flickr_url", "id_y"
            ], inplace=True)
        
        self.all_datapoints = self.all_datapoints.groupby("image_path")
        person_count_mask = self.all_datapoints.size() <= max_num_persons
        self.all_datapoints = self.all_datapoints.aggregate(lambda x: x.tolist())
        self.all_datapoints = self.all_datapoints[person_count_mask]
        self.all_datapoints.reset_index(inplace=True)
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        dp = self.all_datapoints.iloc[idx]
        
        image = read_image(dp["image_path"], mode=ImageReadMode.RGB).to(dtype=T.float32)
        bboxes = dp["bbox"]
        segmentations = dp["segmentation"]
        
        bboxes_ = T.zeros(self.max_num_persons, 4)
        
        bboxes = T.tensor(bboxes, dtype=T.float32).reshape(-1, 2, 2) # ([scale_points(T.tensor(bbox).reshape(2, 2), self.image_shape[::-1], [1, 1]) for bbox in bboxes])
        bboxes = scale_points(bboxes, image.shape[::-1][:2], [1, 1])
        bboxes[:, 1, :] += bboxes[:, 0, :]
        bboxes = bboxes.reshape(-1, 4)
        
        bboxes_[:bboxes.shape[0]] = bboxes[:]
        
        segmentations_ = T.zeros(self.max_num_persons, self.max_segmentation_points, 2)
        for seg in segmentations:
            seg = T.cat([T.tensor(s, dtype=T.float32) for s in seg])
            seg = seg.flatten().reshape(-1, 2)
            seg = seg[:self.max_segmentation_points]
            seg = scale_points(seg, image.shape[::-1][:2], [1, 1])
            seg = pad(seg, (0, 0, 0, max(0, self.max_segmentation_points - seg.shape[0])), value=0)
            segmentations_[:seg.shape[0]] = seg[:]
        
        # # validity mask
        # bbox_validity_mask = T.zeros(self.max_num_persons).bool()
        # bbox_validity_mask[(bboxes != 0).all(-1)] = True
        # segmentation_validity_mask = T.zeros(self.max_num_persons, self.max_segmentation_points).bool()
        # segmentation_validity_mask[(segmentations != 0).all(-1)] = True
        
        # resize full image
        image = resize(image, self.image_shape[::-1]) / 255
        
        # return concatenation of all datapoints
        return image, {
            "bboxes": bboxes_,
            "bboxValidity": 0,
            "segmentations": segmentations_,
            "segmentationValidity": 0
        }

class COCO2017PersonKeypointsDataset(data.Dataset):
    
    def __init__(self, image_folder_path: str, annotation_folder_path: str, image_shape_WH: tuple, min_person_bbox_size: int = 100, crop_padding: int = 20):
        super().__init__()
        
        self.image_shape = image_shape_WH
        self.min_person_bbox_size = min_person_bbox_size
        self.crop_padding = crop_padding
        
        with open(os.path.join(annotation_folder_path, "person_keypoints_train2017.json"), "r") as f:
            train_json = json.load(f)
        with open(os.path.join(annotation_folder_path, "person_keypoints_val2017.json"), "r") as f:
            val_json = json.load(f)
        
        annotations = pd.DataFrame.from_records((*train_json["annotations"], *val_json["annotations"]))
        images = pd.DataFrame.from_records((*train_json["images"], *val_json["images"]))
        
        self.all_datapoints = pd.merge(
            left=annotations,
            right=images, 
            left_on="image_id",
            right_on="id"
        )
        
        # filter on bbox size
        self.all_datapoints = self.all_datapoints[self.all_datapoints["bbox"].apply(lambda x: x[2] > self.min_person_bbox_size and x[3] > self.min_person_bbox_size)]
        
        self.all_datapoints["imagePath"] = image_folder_path + "/" + self.all_datapoints["file_name"]
        
        self.all_datapoints.reset_index(inplace=True)
        
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        dp = self.all_datapoints.iloc[idx]
        
        image = read_image(dp["imagePath"], mode=ImageReadMode.RGB)
        bbox = T.tensor(dp["bbox"], dtype=T.long).reshape(2, 2)
        keypoints = T.tensor(dp["keypoints"], dtype=T.float32).reshape(-1, 3)
        
        bbox[0, :] -= self.crop_padding
        bbox[1, :] += self.crop_padding * 2
        
        person_crop = crop(image, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
        
        keypoints_validity = one_hot((keypoints[:, 2] > 0).to(T.long), 2)
        keypoints_visibility = one_hot((keypoints[:, 2] == 2).to(T.long), 2)
        keypoints = scale_points(keypoints[:, :2] - bbox[0], person_crop.shape[::-1][:2], [1, 1])
        
        bbox = scale_points(bbox, image.shape[::-1][:2], [1, 1])
        bbox[1, :] = bbox[1, :] + bbox[0, :]
        bbox = bbox.flatten()
        
        person_crop = resize(person_crop, self.image_shape[::-1]) / 255
        
        return person_crop, {
            "keypoints": keypoints,
            "keypointValidity": keypoints_validity,
            "keypointVisibility": keypoints_visibility,
        }

class COCOPanopticsObjectDetection(data.Dataset):
    
    def __init__(self, image_folder_path: str, panoptics_path: str, image_shape_WH: tuple, max_number_of_instances: int):
        
        super().__init__()
        
        self.image_shape = image_shape_WH
        self.max_number_of_instances = max_number_of_instances
        
        with open(os.path.join(panoptics_path, "panoptic_train2017.json"), "r") as f:
            train_json = json.load(f)
        with open(os.path.join(panoptics_path, "panoptic_val2017.json"), "r") as f:
            val_json = json.load(f)
        
        annotations = pd.DataFrame.from_records((*train_json["annotations"], *val_json["annotations"]))
        self.images = pd.DataFrame.from_records((*train_json["images"], *val_json["images"]))
        
        # filter on number of segments
        num_segments = annotations["segments_info"].map(len)
        annotations = annotations[(num_segments > 0) & (num_segments <= self.max_number_of_instances)]
        
        # format segment info
        annotations["bboxes"] = annotations["segments_info"].map(lambda x: np.array([xx["bbox"] for xx in x]))
        annotations["categoryIds"] = annotations["segments_info"].map(lambda x: np.array([xx["category_id"] for xx in x]))
        annotations["file_name"] = annotations["file_name"].map(lambda x: f"{x[:-4]}.jpg")
        annotations["imagePath"] = image_folder_path + "/" + annotations["file_name"]
        
        annotations.reset_index(inplace=True)
        self.all_datapoints = annotations
        
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        datapoint = self.all_datapoints.iloc[idx]
        
        image = read_image(datapoint["imagePath"], mode=ImageReadMode.RGB)
        
        bboxes = T.tensor(datapoint["bboxes"], dtype=T.float32).reshape(-1, 2, 2)
        bboxes = scale_points(bboxes, image.shape[::-1][:2], [1, 1])
        bboxes[:, 1, :] += bboxes[:, 0, :]
        bboxes = bboxes.reshape(-1, 4)
        
        categories = one_hot(T.tensor(datapoint["categoryIds"], dtype=T.long), 201)
        
        validity = T.zeros(self.max_number_of_instances, dtype=T.long)
        validity[:len(datapoint["bboxes"])] = 1
        validity = one_hot(validity, 2)
        
        bboxes = pad(bboxes, (0, 0, 0, max(0, self.max_number_of_instances - len(bboxes))), value=0)
        categories = pad(categories, (0, 0, 0, max(0, self.max_number_of_instances - len(categories))), value=0)
        
        image = resize(image, self.image_shape[::-1]) / 255
        
        return image, {
            "validity": validity,
            "bboxes": bboxes,
            "categories": categories
        }

# class COCO2017PanopticsDataset(data.Dataset):
    
#     def __init__(
#         self,
#         image_folder_path: str,
#         panoptics_path: str,
#         output_image_shape_WH: tuple,
#         instance_images_output_shape_WH: tuple,
#         max_number_of_instances: int = 10,
#         center_bbox: bool = True,
#         load_segmentation_masks: bool = True,
#         limit_to_first_n = None):
        
#         super().__init__()
        
#         self.panoptics_path = panoptics_path
#         self.image_folder_path = image_folder_path
        
#         self.output_image_shape = output_image_shape_WH
#         self.instance_images_output_shape = instance_images_output_shape_WH
#         self.max_number_of_instances = max_number_of_instances
#         self.center_bbox = center_bbox
#         self.load_segmentation_masks = load_segmentation_masks
        
        
        
#         val_path = os.path.join(panoptics_path, "panoptic_val2017.json")
#         train_path = os.path.join(panoptics_path, "panoptic_train2017.json")
        
#         # load annotatins and categories
#         with open(train_path, "r") as f:
#             train_json = json.load(f)
#         with open(val_path, "r") as f:
#             val_json = json.load(f)
        
#         # format categories
#         self.categorie_names = {}
#         for cat in train_json["categories"]:
#             self.categorie_names[cat["id"]] = cat["name"]
#         for cat in val_json["categories"]:
#             self.categorie_names[cat["id"]] = cat["name"]
        
#         self.categorie_onehot = {}
#         for cat_i in sorted(list(self.categorie_names.keys())):
#             self.categorie_onehot[cat_i] = one_hot(T.tensor(cat_i), num_classes=max(self.categorie_names.keys()) + 1)
        
#         # load annotations
#         self.all_datapoints = [
#             *zip([True] * len(train_json["annotations"]), train_json["annotations"]), 
#             *zip([False] * len(val_json["annotations"]), val_json["annotations"])
#         ]
        
#         if limit_to_first_n:
#             self.all_datapoints = self.all_datapoints[:limit_to_first_n]
        
#         print("!! WARNING: COCO2017PanopticsDataset segmentation masks contain images not referenced in the image folder, checking will take significantly longer !!")
    
#     def collate_fn_bbox(self, batch):
#         batch = [b for b in batch if b is not None]
#         if len(batch) == 0:
#             return None
        
#         x = T.stack([b["image"] / 255 for b in batch])
#         y = T.stack([(b["bboxes"] / T.tensor([self.output_image_shape] * 2)).flatten(-2) for b in batch])
#         # y = T.stack([b["bboxes"].flatten(-2) for b in batch])
#         v = T.stack([b["instanceValidity"] for b in batch])
#         return x, y, v
    
#     def format_datapoint(self, datapoint, is_train):
#         out = {
#             "originalImagePath": os.path.join(self.image_folder_path, datapoint["file_name"].replace(".png", ".jpg")),
#             "segments": [
#                 {
#                     "category": self.categorie_onehot[segment["category_id"]],
#                     "segmentBbox": T.tensor(segment["bbox"]).reshape(2, 2)
#                 } 
#                 for segment in datapoint["segments_info"]
#             ]
#         }
#         if not self.load_segmentation_masks:
#             return out
        
#         p = "panoptic_train2017" if is_train else "panoptic_val2017"
#         seg_path = os.path.join(self.panoptics_path, p, p, datapoint["file_name"])
#         out["segmentationImagePath"] = seg_path
#         return out
    
#     def get_item_with_segmentation_mask(self, datapoint):
        
#         if len(datapoint["segments"]) == 0:
#             return None
        
#         image = read_image(datapoint["originalImagePath"], mode=ImageReadMode.RGB)
#         segmentation_mask = read_image(datapoint["segmentationImagePath"])
        
#         bboxes = []
#         categories = []
#         instance_images = []
#         instance_mask_images = []
#         for segment in datapoint["segments"]:
            
#             bbox = segment["segmentBbox"]
            
#             instance_image = crop(image, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
#             instance_image = resize(instance_image, self.instance_images_output_shape[::-1])
#             instance_images.append(instance_image)
            
#             instance_mask_image = crop(segmentation_mask, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
#             instance_mask_image = resize(instance_mask_image, self.instance_images_output_shape[::-1])
#             instance_mask_images.append(instance_mask_image)
            
#             # scale bbox and center if needed
#             bbox = scale_points(bbox, image.shape[::-1][:2], self.output_image_shape).to(dtype=T.int16)
#             if self.center_bbox:
#                 bbox = center_bbox(bbox)
#             bboxes.append(bbox)
            
#             categories.append(segment["category"])
        
#         # pad to max number of instances
#         num_instances = len(bboxes)
#         bboxes = T.stack([*bboxes[:self.max_number_of_instances], *[T.zeros_like(bboxes[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
#         categories = T.stack([*categories[:self.max_number_of_instances], *[T.zeros_like(categories[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
#         instance_images = T.stack([*instance_images[:self.max_number_of_instances], *[T.zeros_like(instance_images[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
#         instance_mask_images = T.stack([*instance_mask_images[:self.max_number_of_instances], *[T.zeros_like(instance_mask_images[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
        
#         # create instance validity mask
#         instance_validity = T.zeros(self.max_number_of_instances, dtype=T.bool)
#         instance_validity[:num_instances] = True
        
#         # scale image and segmentation mask
#         image = resize(image, self.output_image_shape[::-1])
#         segmentation_mask = resize(segmentation_mask, self.output_image_shape[::-1])
        
#         return {
#             "image": image,
#             "instanceValidity": instance_validity,
#             "segmentationMask": segmentation_mask,
#             "bboxes": bboxes,
#             "instanceImages": instance_images,
#             "instanceMaskImages": instance_mask_images,
#             "categories": categories
#         }
        
#     def get_item_without_segmentation_mask(self, datapoint):
        
#         if len(datapoint["segments"]) == 0:
#             return None
        
#         image = read_image(datapoint["originalImagePath"], mode=ImageReadMode.RGB)
        
#         bboxes = []
#         categories = []
#         for segment in datapoint["segments"]:
            
#             bbox = segment["segmentBbox"]
            
#             # scale bbox and center if needed
#             bbox = scale_points(bbox, image.shape[::-1][:2], self.output_image_shape).to(dtype=T.int16)
#             if self.center_bbox:
#                 bbox = center_bbox(bbox)
#             bboxes.append(bbox)
            
#             categories.append(segment["category"])
        
#         # pad to max number of instances
#         num_instances = len(datapoint["segments"])
#         bboxes = T.stack([*bboxes[:self.max_number_of_instances], *[T.zeros_like(bboxes[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
#         categories = T.stack([*categories[:self.max_number_of_instances], *[T.zeros_like(categories[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
        
#         # create instance validity mask
#         instance_validity = T.zeros(self.max_number_of_instances, dtype=T.bool)
#         instance_validity[:num_instances] = True
        
#         # scale image and segmentation mask
#         image = resize(image, self.output_image_shape[::-1])
        
#         return {
#             "image": image,
#             "instanceValidity": instance_validity,
#             "bboxes": bboxes,
#             "categories": categories
#         }
    
#     def __len__(self):
#         return len(self.all_datapoints)
    
#     def __getitem__(self, idx):
#         is_train, dp = self.all_datapoints[idx]
#         datapoint = self.format_datapoint(dp, is_train)
        
#         if self.load_segmentation_masks:
#             out = self.get_item_with_segmentation_mask(datapoint)
#         else:
#             out = self.get_item_without_segmentation_mask(datapoint)
        
#         return out

# class COCO2017WholeBodyDataset(data.Dataset):
    
#     def __init__(
#         self,
#         annotations_folder_path: str,
#         image_folder_path: str,
#         full_image_shape_WH: tuple,
#         person_image_shape_WH: tuple,
#         bodypart_image_shape_WH: tuple,
#         min_person_bbox_size: int = 100, # in pixels for both width and height
#         max_number_of_persons: int = 3,
#         load_val_only: bool = False):
        
#         super().__init__()
        
#         self.annotations_folder_path = annotations_folder_path
#         self.image_folder_path = image_folder_path
        
#         self.full_image_shape = full_image_shape_WH
#         self.person_image_shape = person_image_shape_WH
#         self.bodypart_image_shape = bodypart_image_shape_WH
        
#         self.max_number_of_persons = max_number_of_persons
#         self.min_person_bbox_size = min_person_bbox_size
        
#         raw_annotation_datapoints = []
#         raw_image_datapoints = []
#         with open(os.path.join(annotations_folder_path, "coco_wholebody_val_v1.0.json"), "r") as f:
#             j = json.load(f)
#             raw_annotation_datapoints.extend(j["annotations"])
#             raw_image_datapoints.extend(j["images"])
            
#         if not load_val_only:
#             with open(os.path.join(annotations_folder_path, "coco_wholebody_train_v1.0.json"), "r") as f:
#                 j = json.load(f)
#                 raw_annotation_datapoints.extend(j["annotations"])
#                 raw_image_datapoints.extend(j["images"])
        
#         # image id to path map
#         self.image_id_path_map = {dp["id"]: os.path.join(self.image_folder_path, dp["file_name"]) for dp in raw_image_datapoints}
        
#         self.all_datapoints = self.format_datapoints(raw_annotation_datapoints)
    
#     def __len__(self):
#         return len(self.all_datapoints)
    
#     def __getitem__(self, idx):
        
#         datapoint = self.all_datapoints[idx]
        
#         # load image
#         image = read_image(datapoint["imagePath"], mode=ImageReadMode.RGB).to(dtype=T.float32)
#         person_bbox = datapoint["personBbox"]
#         face_bbox = datapoint["faceBbox"]
#         left_hand_bbox = datapoint["leftHandBbox"]
#         right_hand_bbox = datapoint["rightHandBbox"]
        
        
#         # crop persons
#         person_crop = crop(image, person_bbox[0][1], person_bbox[0][0], person_bbox[1][1], person_bbox[1][0])
#         person_crop = resize(person_crop, self.person_image_shape[::-1])
        
#         # gather all keypoints
#         keypoints, visibility, validity = self.concat_keypoints(
#             datapoint["bodyKeypoints"], datapoint["faceKeypoints"], datapoint["leftHandKeypoints"], datapoint["rightHandKeypoints"], datapoint["footKeypoints"],
#             datapoint["bodyKeypointVisibility"], datapoint["faceKeypointVisibility"], datapoint["leftHandKeypointVisibility"], datapoint["rightHandKeypointVisibility"], datapoint["footKeypointVisibility"],
#             datapoint["bodyKeypointValidity"], datapoint["faceKeypointValidity"], datapoint["leftHandKeypointValidity"], datapoint["rightHandKeypointValidity"], datapoint["footKeypointValidity"]
#         )
        
#         # scale keypoints and bounding boxes
#         keypoints = scale_points(keypoints - person_bbox[0], person_bbox[1], [1, 1])
        
#         # scale bounding boxes
#         face_bbox = scale_points(face_bbox - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
#         left_hand_bbox = scale_points(left_hand_bbox - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
#         right_hand_bbox = scale_points(right_hand_bbox - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
        
#         face_bbox[1] = (face_bbox[0] + face_bbox[1]).reshape(-1, 4)
#         left_hand_bbox[1] = (left_hand_bbox[0] + left_hand_bbox[1]).reshape(-1, 4)
#         right_hand_bbox[1] = (right_hand_bbox[0] + right_hand_bbox[1]).reshape(-1, 4)
        
#         return {
#             "personImages": person_crop,
#             "faceBbox": face_bbox,
#             "leftHandBbox": left_hand_bbox,
#             "rightHandBbox": right_hand_bbox,
#             "keypoints": keypoints,
#             "keypointsVisibility": visibility,
#             "keyponitsValidity": validity,
#         }
    
#     # TODO: create a function to formal keypoints back to readable
#     def concat_keypoints(
#         self,
#         body_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints, foot_keypoints,
#         body_visibility, face_visibility, left_hand_visibility, right_hand_visibility, foot_visibility,
#         body_validity, face_validity, left_hand_validity, right_hand_validity, foot_validity):
#         return (
#             T.cat([
#                 body_keypoints,
#                 face_keypoints,
#                 left_hand_keypoints,
#                 right_hand_keypoints,
#                 foot_keypoints
#             ]),
#             T.cat([
#                 body_visibility,
#                 face_visibility,
#                 left_hand_visibility,
#                 right_hand_visibility,
#                 foot_visibility
#             ]),
#             T.cat([
#                 body_validity,
#                 face_validity,
#                 left_hand_validity,
#                 right_hand_validity,
#                 foot_validity
#             ])
#         )
    
#     def format_datapoint(self, datapoint):
        
#         # drop if bbox not large enough
#         if datapoint["bbox"][2] < self.min_person_bbox_size or datapoint["bbox"][3] < self.min_person_bbox_size:
#             return None
        
#         keypoints = T.tensor(datapoint["keypoints"], dtype=T.int16).reshape(-1, 3)
#         face_kpts = T.tensor(datapoint["face_kpts"], dtype=T.int16).reshape(-1, 3)
#         lefthand_kpts = T.tensor(datapoint["lefthand_kpts"], dtype=T.int16).reshape(-1, 3)
#         righthand_kpts = T.tensor(datapoint["righthand_kpts"], dtype=T.int16).reshape(-1, 3)
#         foot_kpts = T.tensor(datapoint["foot_kpts"], dtype=T.int16).reshape(-1, 3)
#         person_bbox = T.tensor(datapoint["bbox"], dtype=T.int16).reshape(2, 2)
#         face_bbox = T.tensor(datapoint["face_box"], dtype=T.int16).reshape(2, 2)
#         left_hand_bmbox = T.tensor(datapoint["lefthand_box"], dtype=T.int16).reshape(2, 2)
#         right_hand_bbox = T.tensor(datapoint["righthand_box"], dtype=T.int16).reshape(2, 2)
#         face_valid = T.tensor(datapoint["face_valid"], dtype=T.bool)
#         lefthand_valid = T.tensor(datapoint["lefthand_valid"], dtype=T.bool)
#         righthand_valid = T.tensor(datapoint["righthand_valid"], dtype=T.bool)
        
#         keypoint_visibility = keypoints[:, 2] == 2
#         face_kpts_visibility = face_kpts[:, 2] == 2
#         lefthand_kpts_visibility = lefthand_kpts[:, 2] == 2
#         righthand_kpts_visibility = righthand_kpts[:, 2] == 2
#         foot_kpts_visibility = foot_kpts[:, 2] == 2
        
#         keypoints_validity = keypoints[:, 2] == 1
#         face_kpts_validity = face_kpts[:, 2] == 1
#         lefthand_kpts_validity = lefthand_kpts[:, 2] == 1
#         righthand_kpts_validity = righthand_kpts[:, 2] == 1
#         foot_kpts_validity = foot_kpts[:, 2] == 1
        
#         keypoints = keypoints[:, :2]
#         face_kpts = face_kpts[:, :2]
#         lefthand_kpts = lefthand_kpts[:, :2]
#         righthand_kpts = righthand_kpts[:, :2]
#         foot_kpts = foot_kpts[:, :2]
        
#         return {
#             "imagePath": self.image_id_path_map[datapoint["image_id"]],
#             "personBbox": person_bbox,
#             "faceBbox": face_bbox,
#             "leftHandBbox": left_hand_bmbox,
#             "rightHandBbox": right_hand_bbox,
#             "bodyKeypoints": keypoints,
#             "bodyKeypointVisibility": keypoint_visibility,
#             "bodyKeypointValidity": keypoints_validity,
#             "faceKeypoints": face_kpts,
#             "faceKeypointVisibility": face_kpts_visibility,
#             "faceKeypointValidity": face_kpts_validity,
#             "leftHandKeypoints": lefthand_kpts,
#             "leftHandKeypointVisibility": lefthand_kpts_visibility,
#             "leftHandKeypointValidity": lefthand_kpts_validity,
#             "rightHandKeypoints": righthand_kpts,
#             "rightHandKeypointVisibility": righthand_kpts_visibility,
#             "rightHandKeypointValidity": righthand_kpts_validity,
#             "footKeypoints": foot_kpts,
#             "footKeypointVisibility": foot_kpts_visibility,
#             "footKeypointValidity": foot_kpts_validity,
#             "faceValidity": face_valid,
#             "leftHandValidity": lefthand_valid,
#             "rightHandValidity": righthand_valid,
#         }
    
#     def format_datapoints(self, annotation_datapoints):
        
#         # format datapoints
#         formatted_datapoints = []
#         for dp in annotation_datapoints:
#             formatted_dp = self.format_datapoint(dp)
#             if formatted_dp is None:
#                 continue
#             formatted_datapoints.append(formatted_dp)
        
#         return formatted_datapoints

class COCO2017PersonWholeBody(data.Dataset):
    
    def __init__(self, annotations_folder_path: str, image_folder_path: str, image_shape_WH: tuple, min_person_bbox_size: int = 100, padding: int = 20):
        super().__init__()
        
        self.annotations_folder_path = annotations_folder_path
        self.image_folder_path = image_folder_path
        self.image_shape = image_shape_WH
        self.padding = padding
        
        area = min_person_bbox_size ** 2
        
        with open(os.path.join(annotations_folder_path, "coco_wholebody_val_v1.0.json"), "r") as f:
            val_json = json.load(f)
        with open(os.path.join(annotations_folder_path, "coco_wholebody_train_v1.0.json"), "r") as f:
            train_json = json.load(f)
        
        images = pd.DataFrame.from_records((*train_json["images"], *val_json["images"]))
        annotations = pd.DataFrame.from_records((*train_json["annotations"], *val_json["annotations"]))
        # images = pd.DataFrame.from_records(val_json["images"])
        # annotations = pd.DataFrame.from_records(val_json["annotations"])
        
        self.all_datapoints = pd.merge(annotations, images, right_on="id", left_on="image_id")
        self.all_datapoints["image_path"] = self.image_folder_path + "/" + self.all_datapoints["file_name"]
        self.all_datapoints = self.all_datapoints[self.all_datapoints["bbox"].map(lambda x: x[2] * x[3] > area)]
        
        validity_mask = (self.all_datapoints["num_keypoints"] != 0) | self.all_datapoints["face_valid"] | self.all_datapoints["lefthand_valid"] | self.all_datapoints["righthand_valid"] | self.all_datapoints["foot_valid"]
        self.all_datapoints = self.all_datapoints[validity_mask]
        
        self.all_datapoints.reset_index(drop=True, inplace=True)
        
    def format_keypoints(self, datapoint):
        kpts = T.cat([
            T.tensor(datapoint["keypoints"]).reshape(-1, 3),
            T.tensor(datapoint["face_kpts"]).reshape(-1, 3),
            T.tensor(datapoint["lefthand_kpts"]).reshape(-1, 3),
            T.tensor(datapoint["righthand_kpts"]).reshape(-1, 3),
            T.tensor(datapoint["foot_kpts"]).reshape(-1, 3)
        ]).to(dtype=T.float32)
        
        kpts_visibility = kpts[:, 2] == 2
        kpts_validity = kpts[:, 2] > 0
        kpts = kpts[:, :2]
        
        return kpts, kpts_validity, kpts_visibility
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        
        datapoint = self.all_datapoints.iloc[idx]
        
        # load image
        image = read_image(datapoint["image_path"], mode=ImageReadMode.RGB).to(dtype=T.float32)
        person_bbox = T.tensor(datapoint["bbox"], dtype=T.int16).reshape(2, 2)
        face_bbox = T.tensor(datapoint["face_box"], dtype=T.int16).reshape(2, 2)
        lefthand_bbox =T.tensor(datapoint["lefthand_box"], dtype=T.int16).reshape(2, 2)
        righthand_bbox = T.tensor(datapoint["righthand_box"], dtype=T.int16).reshape(2, 2)
        all_keypoints, kpt_val, kpt_vis = self.format_keypoints(datapoint)
        
        # add padding to bounding boxes
        person_bbox[0] -= self.padding
        person_bbox[1] += self.padding * 2
        
        # crop persons
        person_crop = crop(image, person_bbox[0][1], person_bbox[0][0], person_bbox[1][1], person_bbox[1][0])
        person_crop = resize(person_crop, self.image_shape[::-1]) / 255
        
        # scale keypoints
        all_keypoints = scale_points(all_keypoints - person_bbox[0], person_bbox[1], [1, 1])
        
        # scale bounding boxes
        face_bbox = scale_points(face_bbox.to(dtype=T.float32) - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
        lefthand_bbox = scale_points(lefthand_bbox.to(dtype=T.float32) - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
        righthand_bbox = scale_points(righthand_bbox.to(dtype=T.float32) - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
        
        # format bounding boxes
        person_bbox[1] += person_bbox[0]
        face_bbox[1] += face_bbox[0]
        lefthand_bbox[1] += lefthand_bbox[0]
        righthand_bbox[1] += righthand_bbox[0]
        
        # onehot encode keypoints
        kpt_val = one_hot(kpt_val.to(dtype=T.int64), 2)
        kpt_vis = one_hot(kpt_vis.to(dtype=T.int64), 2)
        
        return person_crop, {
            "keypoints": all_keypoints,
            "keypointsValidity": kpt_val,
            "keypointsVisibility": kpt_vis,
            "faceBbox": face_bbox.flatten(),
            "lefthandBbox": lefthand_bbox.flatten(),
            "righthandBbox": righthand_bbox.flatten(),
        }
    
    # TODO: create a function to formal keypoints back to readable
    def concat_keypoints(
        self,
        body_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints, foot_keypoints,
        body_visibility, face_visibility, left_hand_visibility, right_hand_visibility, foot_visibility,
        body_validity, face_validity, left_hand_validity, right_hand_validity, foot_validity):
        return (
            T.cat([body_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints, foot_keypoints]),
            T.cat([body_visibility, face_visibility, left_hand_visibility, right_hand_visibility, foot_visibility]),
            T.cat([body_validity, face_validity, left_hand_validity, right_hand_validity, foot_validity])
        )


class HAKELarge(data.Dataset):
    def __init__(self, annotation_path, image_path, image_shape_WH, max_num_bboxes=10):
        super().__init__()
        
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.image_shape = image_shape_WH
        self.max_num_bboxes = max_num_bboxes
        
        with open(os.path.join(annotation_path, "hake_large_annotation.json"), "r") as f:
            datapoints = json.load(f)
        
        self.all_datapoints = []
        for k in datapoints:
            formatted_datapoints = self.format_datapoint(k, datapoints[k])
            if formatted_datapoints is not None:
                self.all_datapoints.append(formatted_datapoints)
    
    def format_datapoint(self, k, v):
        
        if len(v["labels"]) == 0:
            return None
        
        if v["dataset"] == "vcoco":
            return None
        
        return {
            "imagePath": os.path.join(self.image_path, v["dataset"], k),
            "humanBboxes": T.stack([T.tensor(lab["human_bbox"]) for lab in v["labels"]]).to(dtype=T.float32).reshape(-1, 2, 2),
            "objectBboxes": T.stack([T.tensor(lab["object_bbox"]) for lab in v["labels"]]).to(dtype=T.float32).reshape(-1, 2, 2),
            # "actionLabels": [{
            #     "humanPart": T.tensor([act_lab["human_part"] for act_lab in lab["action_labels"]]).to(dtype=T.float32),
            #     "partState": T.tensor([act_lab["partstate"] for act_lab in lab["action_labels"]]).to(dtype=T.float32),
            # } for lab in v["labels"] if "action_labels" in lab and len(lab["action_labels"]) > 0]
        }
        
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        
        dp = self.all_datapoints[idx]
        
        image = read_image(dp["imagePath"], ImageReadMode.RGB) / 255
        
        human_bboxes = scale_points(dp["humanBboxes"], image.shape[1:][::-1], [1, 1])
        human_bboxes = human_bboxes.reshape(-1, 4)
        human_bboxes = pad(human_bboxes, (0, 0, 0, max(0, self.max_num_bboxes - human_bboxes.shape[0])), value=0)
        
        object_bboxes = scale_points(dp["objectBboxes"], image.shape[1:][::-1], [1, 1])
        object_bboxes = object_bboxes.reshape(-1, 4)
        object_bboxes = pad(object_bboxes, (0, 0, 0, max(0, self.max_num_bboxes - object_bboxes.shape[0])), value=0)
        
        
        image = resize(image, self.image_shape[::-1])
        
        return image, dp, {
            "humanBboxes": human_bboxes,
            "objectBboxes": object_bboxes
        }
