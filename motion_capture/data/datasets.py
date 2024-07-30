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
import torchvision.transforms as TVTransform

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
    
    if points.shape[1] == 2:
        return points / T.tensor(input_shape, dtype=T.float32) * T.tensor(output_shape, dtype=T.float32)
    else:
        return T.cat([points[:, :2] / T.tensor(input_shape) * T.tensor(output_shape), points[:, 2:]], dim=1)

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
    
# def norm_image(image):
#     return image / 255

# y = T.stack([(b[1] / T.tensor([self.image_shape] * 2).flatten(-2)) for b in batch]).to(dtype=T.float32)
# v = T.stack([b[2] for b in batch]).to(dtype=T.float32)
# return x, y, v

# def scale_bbox(bbox: list, current_image_shape: list, new_image_shape: list) -> list:
#     '''
#         current_image_shape and new_image_shape in format [W, H]
#     '''
    
#     x1, y1, w, h = bbox
    
#     x1 = x1 / current_image_shape[0] * new_image_shape[0]
#     w = w / current_image_shape[0] * new_image_shape[0]
#     y1 = y1 / current_image_shape[1] * new_image_shape[1]
#     h = h / current_image_shape[1] * new_image_shape[1]
    
#     return [x1, y1, w, h]

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


class WIDERFaceFaceDetection(data.Dataset):
    
    def __init__(self, path: str, image_shape_WH: tuple, max_number_of_faces: int):
        
        super(type(self), self).__init__()
        
        self.image_shape = image_shape_WH
        self.max_number_of_faces = max_number_of_faces
        
        train_annotations = self.extract_formated_datapoints(
            anno_file_path=os.path.join(path, "train", "train_bbx_gt.txt"),
            image_file_path=os.path.join(path, "train", "images")
        )
        val_annotations = self.extract_formated_datapoints(
            anno_file_path=os.path.join(path, "val", "val_bbx_gt.txt"),
            image_file_path=os.path.join(path, "val", "images")
        )
        
        self.annotation_datapoints = deepcopy([*train_annotations, *val_annotations])
    
    def __len__(self):
        return len(self.annotation_datapoints)
    
    def __getitem__(self, idx):
        
        datapoint = self.annotation_datapoints[idx]
        
        image = read_image(datapoint["imagePath"], mode=ImageReadMode.RGB)
        
        # get bounding boxes and scale them
        bboxes = []
        for i in range(min(datapoint["numberOfFaces"], self.max_number_of_faces)):
            bbox = datapoint["faces"][i]["faceBbox"]
            bbox = scale_points(bbox, image.shape[::-1][:2], [1, 1])
            bbox[1, :] = bbox[0, :] + bbox[1, :]
            bboxes.append(bbox)
        
        validity = T.zeros(self.max_number_of_faces).bool()
        validity[:datapoint["numberOfFaces"]] = True
        
        # pad bboxes if there are less than max_number_of_faces
        bboxes = pad(T.stack(bboxes).reshape(-1, 4)[:self.max_number_of_faces], (0, 0, 0, max(0, self.max_number_of_faces - len(bboxes))), value=0)
        
        # resize image
        image = resize(image, self.image_shape[::-1], antialias=True) / 255
        
        return image, {
            "bboxes": bboxes, 
            "bboxValidity": validity
        }
    
    def extract_formated_datapoints(self, anno_file_path: str, image_file_path: str):
        formated_datapoints = []
        with open(anno_file_path, "r") as f:
            
            fulltext = "".join(f.readlines())
            datapoints = re.split("(.*--.*jpg\n)", fulltext)
            
            for (image_info, face_info) in zip(datapoints[1:-1:2], datapoints[2::2]):
                face_info = face_info.split("\n")[:-1]
                
                image_folder, image_name = image_info.split("/")
                image_name = image_name[:-1] # to remove \n
                number_of_faces = int(face_info[0])
                
                if number_of_faces > self.max_number_of_faces:
                    continue
                
                faces = face_info[1:]
                formated_faces = []
                
                for face in faces:
                    face = face.split(" ")[:-1]
                    
                    formated_faces.append({
                        "faceBbox": T.tensor([int(x) for x in face[:4]]).reshape(-1, 2),
                        "indicators": [int(x) for x in face[4:]]
                    })
                
                if len(faces) == 0:
                    continue
                
                formated_datapoints.append({
                    "imagePath": os.path.join(image_file_path, image_folder, image_name),
                    "numberOfFaces": len(faces),
                    "faces": formated_faces
                })
        return formated_datapoints

class WFLWFaceDetection(data.Dataset):
    def __init__(self, image_shape_WH: tuple, path: str, max_number_of_faces: int):
        super(type(self), self).__init__()
        
        self.max_number_of_faces = max_number_of_faces
        self.image_shape = image_shape_WH
        self.image_folder_path = os.path.join(path, "images")
        
        with open(os.path.join(path, "annotations", "train.txt")) as f:
            train_datapoints = self.extract_formatted_datapoints(f.readlines())
        with open(os.path.join(path, "annotations", "validation.txt")) as f:
            validation_datapoints = self.extract_formatted_datapoints(f.readlines())
        
        # group datapoints by image path
        all_datapoints_by_image = {}
        for dp in (*train_datapoints, *validation_datapoints):
            if dp["imagePath"] not in all_datapoints_by_image:
                all_datapoints_by_image[dp["imagePath"]] = []
            all_datapoints_by_image[dp["imagePath"]].append(dp)
        
        self.all_datapoints = [dp for dp in all_datapoints_by_image.values() if len(dp) <= max_number_of_faces]
    
    def extract_formatted_datapoints(self, raw_lines: list):
        individual_datapoints = []
        for l in raw_lines:
            line = l.split(" ")
            bbox = [int(ell) for ell in line[196:200]]
            bbox = [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
            individual_datapoints.append({
                "imagePath": os.path.join(self.image_folder_path, line[-1])[:-1],
                "bbox": T.tensor(bbox, dtype=T.int16).reshape(2, 2)
            })
        
        return individual_datapoints
    
    def __len__(self):
        return (len(self.all_datapoints))
    
    def __getitem__(self, idx):
        datapoints = self.all_datapoints[idx][:self.max_number_of_faces]
        
        if len(datapoints) == 0:
            return None
        
        full_image = read_image(datapoints[0]["imagePath"], mode=ImageReadMode.RGB)
        
        bboxes = []
        for datapoint in datapoints:
            bbox = scale_points(datapoint["bbox"], full_image.shape[::-1][:2], [1, 1]).to(dtype=T.float32)
            bbox[1, :] = bbox[0, :] + bbox[1, :]
            bboxes.append(bbox)
        
        validity = T.zeros(self.max_number_of_faces).bool()
        validity[:len(bboxes)] = True
        
        # stack all and shuffle
        bboxes = pad(T.stack(bboxes).reshape(-1, 4), (0, 0, 0, max(0, self.max_number_of_faces - len(bboxes))), value=0)
        
        # resize full image
        image = resize(full_image, self.image_shape[::-1], antialias=True) / 255
        
        return image, {
            "bboxes": bboxes, 
            "bboxValidity": validity
        }

class COFWFaceDetection(data.Dataset):
    def __init__(self, path: str, image_shape_WH: tuple):
        super(type(self), self).__init__()
        
        self.image_shape = image_shape_WH
        
        train_datapoints = self.extract_formatted_datapoints(os.path.join(path, "color_train.mat"))
        val_datapoints = self.extract_formatted_datapoints(os.path.join(path, "color_test.mat"))
        
        self.all_datapoints = [*train_datapoints, *val_datapoints]
    
    def extract_formatted_datapoints(self, path: str):
        file = h5py.File(path, "r")
        keys = list(file.get("/"))
        
        image_refs = np.array(file.get(keys[1])).squeeze()
        bboxes = T.tensor(np.array(file.get(keys[2])), dtype=T.int16).squeeze().T.reshape(-1, 2, 2)
        images = (np.array(file[img_ref]) for img_ref in image_refs)
        return zip(images, bboxes)
        
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        
        datapoint = self.all_datapoints[idx]
        
        # load image
        image = T.tensor(datapoint[0], dtype=T.uint8)
        
        if len(image.shape) == 2:
            return None
        
        image = image.permute(0, 2, 1)
        
        # scale keypoints and bbox acording to the new full image size
        bbox = scale_points(datapoint[1], image.shape[::-1][:2], [1, 1])
        bbox[1, :] = bbox[0, :] + bbox[1, :]
        bbox = bbox.reshape(4)
        
        # scale full image and face image
        image = resize(image, self.image_shape[::-1], antialias=True) / 255
        
        return image, {
            "bbox": bbox
        }


# class WFLWFaceKeypointDetection(data.Dataset):
#     def __init__(self, image_shape_WH: tuple, path: str):
#         super(type(self), self).__init__()
        
#         self.image_shape = image_shape_WH
#         self.image_folder_path = os.path.join(path, "images")
        
#         with open(os.path.join(path, "annotations", "train.txt")) as f:
#             train_datapoints = self.extract_formatted_datapoints(f.readline())
#         with open(os.path.join(path, "annotations", "validation.txt")) as f:
#             validation_datapoints = self.extract_formatted_datapoints(f.readline())
        
#         all_datapoints = [*train_datapoints, *validation_datapoints]
        
#         self.all_datapoints = deepcopy(all_datapoints)
        
#         # # group datapoints by image path
#         # all_datapoints_by_image = {}
#         # for dp in all_datapoints:
#         #     if dp["imagePath"] not in all_datapoints_by_image:
#         #         all_datapoints_by_image[dp["imagePath"]] = []
#         #     all_datapoints_by_image[dp["imagePath"]].append(dp)
        
#         # self.all_datapoints = deepcopy(list(all_datapoints_by_image.values()))
    
#     def __len__(self):
#         return (len(self.all_datapoints))
    
#     def __getitem__(self, idx):
        
#         datapoint = self.all_datapoints[idx]
#         full_image = read_image(datapoint["imagePath"], mode=ImageReadMode.RGB)
#         bbox = datapoint["bbox"]
#         keypoints = datapoint["keypoints"]
#         indicators = datapoint["indicators"]
        
#         face_image = crop(full_image, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
#         scaled_keypoints = scale_points(keypoints - bbox[0], face_image.shape[::-1][:2], self.image_shape)
#         face_image = resize(face_image, self.image_shape[::-1], antialias=True)
        
#         return face_image, scaled_keypoints, indicators
        
#     def extract_formatted_datapoints(self, path):
#         individual_datapoints = []
#         with open(path) as f:
#             for l in f.readlines():
#                 line = l.split(" ")
                
#                 bbox = [int(ell) for ell in line[196:200]]
#                 bbox = [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
                
#                 individual_datapoints.append({
#                     "imagePath": os.path.join(self.image_folder_path, line[-1])[:-1],
#                     "keypoints": T.tensor([float(a) for a in line[:196]]).reshape(-1, 2),
#                     "bbox": T.tensor(bbox, dtype=T.int16).reshape(2, 2),
#                     "indicators": T.tensor([int(ell) for ell in line[200:206]])
#                 })
        
#         return individual_datapoints


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
        max_segmentation_points: int = 100):
        
        super().__init__()
        
        self.max_num_persons = max_num_persons
        self.max_segmentation_points = max_segmentation_points
        self.image_shape = image_shape_WH
        
        with open(os.path.join(annotation_folder_path, "person_keypoints_train2017.json"), "r") as f:
            train_json = json.load(f)
        with open(os.path.join(annotation_folder_path, "person_keypoints_val2017.json"), "r") as f:
            val_json = json.load(f)
        
        self.image_path_map = {}
        for image in train_json["images"]:
            self.image_path_map[image["id"]] = os.path.join(image_folder_path, image["file_name"])
        for image in val_json["images"]:
            self.image_path_map[image["id"]] = os.path.join(image_folder_path, image["file_name"])
        
        self.all_datapoints = {}
        for annotation in (*train_json["annotations"], *val_json["annotations"]):
            if annotation["image_id"] not in self.image_path_map:
                continue
            
            formatted_annotation = self.format_datapoint(annotation)
            if formatted_annotation is None:
                continue
            
            if annotation["image_id"] not in self.all_datapoints:
                self.all_datapoints[annotation["image_id"]] = []
            self.all_datapoints[annotation["image_id"]].append(formatted_annotation)
        
        keys = list(self.all_datapoints.keys())
        for k in keys:
            if len(self.all_datapoints[k]) > max_num_persons:
                self.all_datapoints.pop(k)
        
        self.all_datapoints = list(self.all_datapoints.values())
    
    def format_datapoint(self, datapoint):
        
        # remove etries with multiple segmentations or if not list
        if type(datapoint["segmentation"]) != list:
            return None
        if len(datapoint["segmentation"]) == 0 or len(datapoint["segmentation"]) > 1:
            return None
        
        segmentation = T.tensor(datapoint["segmentation"][0]).reshape(-1, 2)
        
        if segmentation.shape[0] > self.max_segmentation_points:
            return None
        
        return {
            "imagePath": self.image_path_map[datapoint["image_id"]],
            "bbox": T.tensor(datapoint["bbox"], dtype=T.int16).reshape(2, 2),
            "segmentation": segmentation,
        }
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        datapoints = self.all_datapoints[idx]
        
        full_image = read_image(datapoints[0]["imagePath"], mode=ImageReadMode.RGB).to(dtype=T.float32)
        
        bboxes = []
        segmentations = []
        for datapoint in datapoints:
            bbox = datapoint["bbox"]
            segmentation = datapoint["segmentation"]
            
            # resize bbox, keypoints and segmentation to now full image
            full_scaled_bbox = scale_points(bbox, full_image.shape[::-1][:2], [1, 1])
            full_scaled_segmentation = scale_points(segmentation, full_image.shape[::-1][:2], [1, 1])
            full_scaled_segmentation = pad(full_scaled_segmentation[:self.max_segmentation_points], (0, 0, 0, max(0, self.max_segmentation_points - full_scaled_segmentation.shape[0])), value=0)
            
            full_scaled_bbox[1, :] = full_scaled_bbox[0, :] + full_scaled_bbox[1, :]
            
            bboxes.append(full_scaled_bbox)
            segmentations.append(full_scaled_segmentation)
        
        # padding
        bboxes = pad(T.stack(bboxes)[:self.max_num_persons].reshape(-1, 4), (0, 0, 0, max(0, self.max_num_persons - len(bboxes))), value=0)
        segmentations = pad(T.stack(segmentations)[:self.max_num_persons], (0, 0, 0, 0, 0, max(0, self.max_num_persons - len(segmentations))), value=0)
        
        # validity mask
        bbox_validity_mask = T.zeros(self.max_num_persons).bool()
        bbox_validity_mask[(bboxes != 0).all(-1)] = True
        segmentation_validity_mask = T.zeros(self.max_num_persons, self.max_segmentation_points).bool()
        segmentation_validity_mask[(segmentations != 0).all(-1)] = True
        
        # resize full image
        full_image = resize(full_image, self.image_shape[::-1]) / 255
        
        # return concatenation of all datapoints
        return full_image, {
            "bboxes": bboxes,
            "bboxValidity": bbox_validity_mask,
            "segmentations": segmentations,
            "segmentationValidity": segmentation_validity_mask
        }

class COCO2017PersonKeypointsDataset(data.Dataset):
    # ! this is the same set of images produced by COCO2017FullBody
    
    def __init__(self, image_folder_path: str, annotation_folder_path: str, image_shape_WH: tuple, min_person_bbox_size: int = 100, padding: int = 20):
        super().__init__()
        
        self.image_shape = image_shape_WH
        self.min_person_bbox_size = min_person_bbox_size
        self.padding = padding
        
        with open(os.path.join(annotation_folder_path, "person_keypoints_train2017.json"), "r") as f:
            train_json = json.load(f)
        with open(os.path.join(annotation_folder_path, "person_keypoints_val2017.json"), "r") as f:
            test_json = json.load(f)
        
        # get all image_id : image_path pairs
        self.image_path_map = {}
        for image in (*train_json["images"], *test_json["images"]):
            self.image_path_map[image["id"]] = os.path.join(image_folder_path, image["file_name"])
        
        # gruop annotations by image_id
        self.all_datapoints = []
        for annotation in (*train_json["annotations"], *test_json["annotations"]):
            formatted_annotation = self.format_datapoint(annotation)
            if formatted_annotation is None:
                continue
            self.all_datapoints.append(formatted_annotation)
        
    def format_datapoint(self, datapoint):
        
        # remove too large bounding boxes
        if datapoint["bbox"][2] < self.min_person_bbox_size or datapoint["bbox"][3] < self.min_person_bbox_size:
            return None
        
        keypoints = T.tensor(datapoint["keypoints"]).reshape(-1, 3)
        
        return {
            "imagePath": self.image_path_map[datapoint["image_id"]],
            "bbox": T.tensor(datapoint["bbox"], dtype=T.int16).reshape(2, 2),
            "keypoints": keypoints[:, :2],
            "keypointVisibility": keypoints[:, 2] == 2,
            "keypointValidity": keypoints[:, 2] > 0,
        }
        
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        dp = self.all_datapoints[idx]
        
        full_image = read_image(dp["imagePath"], mode=ImageReadMode.RGB).to(dtype=T.float32)
        
        bbox = dp["bbox"]
        keypoints = dp["keypoints"]
        visibility = dp["keypointVisibility"].bool()
        validity = dp["keypointValidity"].bool()
        
        bbox[0, :] -= self.padding
        bbox[1, :] += self.padding * 2
        
        # crop person image and scale keypoints
        person_image = crop(full_image, bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])
        local_scaled_keypoints = scale_points(keypoints - bbox[0], person_image.shape[::-1][:2], [1, 1])
        
        # resize person image
        person_image = resize(person_image, self.image_shape[::-1]) / 255
        
        # return concatenation of all datapoints
        return person_image, {
            "keypoints": local_scaled_keypoints,
            "keypointValidity": validity,
            "keypointVisibility": visibility,
        }

class COCOPanopticsObjectDetection(data.Dataset):
    
    def __init__(
        self,
        image_folder_path: str,
        panoptics_path: str,
        image_shape_WH: tuple,
        max_number_of_instances: int = 10,
        center_bbox: bool = True):
        
        super().__init__()
        
        self.panoptics_path = panoptics_path
        self.image_folder_path = image_folder_path
        
        self.image_shape = image_shape_WH
        self.max_number_of_instances = max_number_of_instances
        self.center_bbox = center_bbox
        
        train_path = os.path.join(panoptics_path, "panoptic_train2017.json")
        val_path = os.path.join(panoptics_path, "panoptic_val2017.json")
        
        # load annotatins and categories
        with open(train_path, "r") as f:
            train_json = json.load(f)
        with open(val_path, "r") as f:
            val_json = json.load(f)
        
        # format categories
        self.categorie_names = {}
        for cat in train_json["categories"]:
            self.categorie_names[cat["id"]] = cat["name"]
        for cat in val_json["categories"]:
            self.categorie_names[cat["id"]] = cat["name"]
        
        self.categorie_onehot = {}
        for cat_i in sorted(list(self.categorie_names.keys())):
            self.categorie_onehot[cat_i] = one_hot(T.tensor(cat_i), num_classes=max(self.categorie_names.keys()) + 1)
        
        # load annotations
        self.all_datapoints = [
            *zip([True] * len(train_json["annotations"]), train_json["annotations"]), 
            *zip([False] * len(val_json["annotations"]), val_json["annotations"])
        ]
    
    def format_datapoint(self, datapoint, is_train):
        return {
            "originalImagePath": os.path.join(self.image_folder_path, datapoint["file_name"].replace(".png", ".jpg")),
            "segments": [
                {
                    "category": self.categorie_onehot[segment["category_id"]],
                    "segmentBbox": T.tensor(segment["bbox"]).reshape(2, 2)
                } 
                for segment in datapoint["segments_info"]
            ]
        }
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        is_train, dp = self.all_datapoints[idx]
        datapoint = self.format_datapoint(dp, is_train)
        
        if len(datapoint["segments"]) == 0:
            return None
        
        image = read_image(datapoint["originalImagePath"], mode=ImageReadMode.RGB)
        
        bboxes = []
        categories = []
        for segment in datapoint["segments"]:
            
            bbox = segment["segmentBbox"]
            
            # scale bbox and center if needed
            bbox = scale_points(bbox, image.shape[::-1][:2], self.image_shape).to(dtype=T.int16)
            if self.center_bbox:
                bbox = center_bbox(bbox)
            bboxes.append(bbox)
            
            categories.append(segment["category"])
        
        # pad to max number of instances
        num_instances = len(datapoint["segments"])
        bboxes = T.stack([*bboxes[:self.max_number_of_instances], *[T.zeros_like(bboxes[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
        categories = T.stack([*categories[:self.max_number_of_instances], *[T.zeros_like(categories[0]) for _ in range(max(0, self.max_number_of_instances - num_instances))]])
        
        # create instance validity mask
        instance_validity = T.zeros(self.max_number_of_instances, dtype=T.int)
        instance_validity[:num_instances] = 1
        
        # scale image and segmentation mask
        image = resize(image, self.image_shape[::-1])
        
        return image, bboxes, instance_validity

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

class COCO2017WholeBodyDataset(data.Dataset):
    
    def __init__(
        self,
        annotations_folder_path: str,
        image_folder_path: str,
        full_image_shape_WH: tuple,
        person_image_shape_WH: tuple,
        bodypart_image_shape_WH: tuple,
        min_person_bbox_size: int = 100, # in pixels for both width and height
        max_number_of_persons: int = 3,
        center_bbox: bool = True,
        load_val_only: bool = False):
        
        super().__init__()
        
        self.annotations_folder_path = annotations_folder_path
        self.image_folder_path = image_folder_path
        
        self.full_image_shape = full_image_shape_WH
        self.person_image_shape = person_image_shape_WH
        self.bodypart_image_shape = bodypart_image_shape_WH
        
        self.center_bbox = center_bbox
        self.max_number_of_persons = max_number_of_persons
        self.min_person_bbox_size = min_person_bbox_size
        
        raw_annotation_datapoints = []
        raw_image_datapoints = []
        with open(os.path.join(annotations_folder_path, "coco_wholebody_val_v1.0.json"), "r") as f:
            j = json.load(f)
            raw_annotation_datapoints.extend(j["annotations"])
            raw_image_datapoints.extend(j["images"])
            
        if not load_val_only:
            with open(os.path.join(annotations_folder_path, "coco_wholebody_train_v1.0.json"), "r") as f:
                j = json.load(f)
                raw_annotation_datapoints.extend(j["annotations"])
                raw_image_datapoints.extend(j["images"])
        
        # image id to path map
        self.image_id_path_map = {dp["id"]: os.path.join(self.image_folder_path, dp["file_name"]) for dp in raw_image_datapoints}
        
        self.all_datapoints = self.format_datapoints(raw_annotation_datapoints)
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        
        
        datapoint = self.all_datapoints[idx]
        
        # load image
        image = read_image(datapoint["imagePath"], mode=ImageReadMode.RGB).to(dtype=T.float32)
        person_bbox = datapoint["personBbox"]
        face_bbox = datapoint["faceBbox"]
        left_hand_bbox = datapoint["leftHandBbox"]
        right_hand_bbox = datapoint["rightHandBbox"]
        
        
        # crop persons
        person_crop = crop(image, person_bbox[0][1], person_bbox[0][0], person_bbox[1][1], person_bbox[1][0])
        person_crop = resize(person_crop, self.person_image_shape[::-1])
        
        # gather all keypoints
        keypoints, visibility, validity = self.concat_keypoints(
            datapoint["bodyKeypoints"], datapoint["faceKeypoints"], datapoint["leftHandKeypoints"], datapoint["rightHandKeypoints"], datapoint["footKeypoints"],
            datapoint["bodyKeypointVisibility"], datapoint["faceKeypointVisibility"], datapoint["leftHandKeypointVisibility"], datapoint["rightHandKeypointVisibility"], datapoint["footKeypointVisibility"],
            datapoint["bodyKeypointValidity"], datapoint["faceKeypointValidity"], datapoint["leftHandKeypointValidity"], datapoint["rightHandKeypointValidity"], datapoint["footKeypointValidity"]
        )
        
        # scale keypoints and bounding boxes
        keypoints = scale_points(keypoints - person_bbox[0], person_bbox[1], self.person_image_shape)
        
        # scale bounding boxes
        face_bbox = scale_points(face_bbox - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], self.person_image_shape)
        left_hand_bbox = scale_points(left_hand_bbox - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], self.person_image_shape)
        right_hand_bbox = scale_points(right_hand_bbox - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], self.person_image_shape)
        
        if self.center_bbox:
            face_bbox = center_bbox(face_bbox)
            left_hand_bbox = center_bbox(left_hand_bbox)
            right_hand_bbox = center_bbox(right_hand_bbox)
        
        return {
            "personImages": person_crop,
            "keypoints": keypoints,
            "keypointsVisibility": visibility,
            "keyponitsValidity": validity,
            "faceBbox": face_bbox,
            "leftHandBbox": left_hand_bbox,
            "rightHandBbox": right_hand_bbox
        }
    
    # TODO: create a function to formal keypoints back to readable
    def concat_keypoints(
        self,
        body_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints, foot_keypoints,
        body_visibility, face_visibility, left_hand_visibility, right_hand_visibility, foot_visibility,
        body_validity, face_validity, left_hand_validity, right_hand_validity, foot_validity):
        return (
            T.cat([
                body_keypoints,
                face_keypoints,
                left_hand_keypoints,
                right_hand_keypoints,
                foot_keypoints
            ]),
            T.cat([
                body_visibility,
                face_visibility,
                left_hand_visibility,
                right_hand_visibility,
                foot_visibility
            ]),
            T.cat([
                body_validity,
                face_validity,
                left_hand_validity,
                right_hand_validity,
                foot_validity
            ])
        )
    
    def format_datapoint(self, datapoint):
        
        # drop if bbox not large enough
        if datapoint["bbox"][2] < self.min_person_bbox_size or datapoint["bbox"][3] < self.min_person_bbox_size:
            return None
        
        keypoints = T.tensor(datapoint["keypoints"], dtype=T.int16).reshape(-1, 3)
        face_kpts = T.tensor(datapoint["face_kpts"], dtype=T.int16).reshape(-1, 3)
        lefthand_kpts = T.tensor(datapoint["lefthand_kpts"], dtype=T.int16).reshape(-1, 3)
        righthand_kpts = T.tensor(datapoint["righthand_kpts"], dtype=T.int16).reshape(-1, 3)
        foot_kpts = T.tensor(datapoint["foot_kpts"], dtype=T.int16).reshape(-1, 3)
        person_bbox = T.tensor(datapoint["bbox"], dtype=T.int16).reshape(2, 2)
        face_bbox = T.tensor(datapoint["face_box"], dtype=T.int16).reshape(2, 2)
        left_hand_bmbox = T.tensor(datapoint["lefthand_box"], dtype=T.int16).reshape(2, 2)
        right_hand_bbox = T.tensor(datapoint["righthand_box"], dtype=T.int16).reshape(2, 2)
        face_valid = T.tensor(datapoint["face_valid"], dtype=T.bool)
        lefthand_valid = T.tensor(datapoint["lefthand_valid"], dtype=T.bool)
        righthand_valid = T.tensor(datapoint["righthand_valid"], dtype=T.bool)
        
        keypoint_visibility = keypoints[:, 2] == 2
        face_kpts_visibility = face_kpts[:, 2] == 2
        lefthand_kpts_visibility = lefthand_kpts[:, 2] == 2
        righthand_kpts_visibility = righthand_kpts[:, 2] == 2
        foot_kpts_visibility = foot_kpts[:, 2] == 2
        
        keypoints_validity = keypoints[:, 2] == 1
        face_kpts_validity = face_kpts[:, 2] == 1
        lefthand_kpts_validity = lefthand_kpts[:, 2] == 1
        righthand_kpts_validity = righthand_kpts[:, 2] == 1
        foot_kpts_validity = foot_kpts[:, 2] == 1
        
        keypoints = keypoints[:, :2]
        face_kpts = face_kpts[:, :2]
        lefthand_kpts = lefthand_kpts[:, :2]
        righthand_kpts = righthand_kpts[:, :2]
        foot_kpts = foot_kpts[:, :2]
        
        return {
            "imagePath": self.image_id_path_map[datapoint["image_id"]],
            "personBbox": person_bbox,
            "faceBbox": face_bbox,
            "leftHandBbox": left_hand_bmbox,
            "rightHandBbox": right_hand_bbox,
            "bodyKeypoints": keypoints,
            "bodyKeypointVisibility": keypoint_visibility,
            "bodyKeypointValidity": keypoints_validity,
            "faceKeypoints": face_kpts,
            "faceKeypointVisibility": face_kpts_visibility,
            "faceKeypointValidity": face_kpts_validity,
            "leftHandKeypoints": lefthand_kpts,
            "leftHandKeypointVisibility": lefthand_kpts_visibility,
            "leftHandKeypointValidity": lefthand_kpts_validity,
            "rightHandKeypoints": righthand_kpts,
            "rightHandKeypointVisibility": righthand_kpts_visibility,
            "rightHandKeypointValidity": righthand_kpts_validity,
            "footKeypoints": foot_kpts,
            "footKeypointVisibility": foot_kpts_visibility,
            "footKeypointValidity": foot_kpts_validity,
            "faceValidity": face_valid,
            "leftHandValidity": lefthand_valid,
            "rightHandValidity": righthand_valid,
        }
    
    def format_datapoints(self, annotation_datapoints):
        
        # format datapoints
        formatted_datapoints = []
        for dp in annotation_datapoints:
            formatted_dp = self.format_datapoint(dp)
            if formatted_dp is None:
                continue
            formatted_datapoints.append(formatted_dp)
        
        return formatted_datapoints

class HAKELarge(data.Dataset):
    def __init__(self, annotation_path, image_path, image_shape_WH):
        super().__init__()
        
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.image_shape = image_shape_WH
        
        with open(os.path.join(annotation_path, "hake_large_annotation.json"), "r") as f:
            self.datapoints = json.load(f)
        
        
    def __len__(self):
        return len(self.datapoints)
    
