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
from torch.utils import data
from torchvision.transforms.functional import resize, crop
from torchvision.io import read_image

# ------------------------------------------------------------------------

# Segmentation for classification: 
# - https://github.com/qubvel/segmentation_models.pytorch

# ------------------------------------------------------------------------

def scale_bbox(bbox: list, current_image_shape: list, new_image_shape: list) -> list:
    '''
        current_image_shape and new_image_shape in format [W, H]
    '''
    
    x1, y1, w, h = bbox
    
    x1 = x1 / current_image_shape[0] * new_image_shape[0]
    w = w / current_image_shape[0] * new_image_shape[0]
    y1 = y1 / current_image_shape[1] * new_image_shape[1]
    h = h / current_image_shape[1] * new_image_shape[1]
    
    return [x1, y1, w, h]

# ------------------------------------------------------------------------

class WIDERFaceDataset(data.Dataset):
    
    def __init__(self,
        output_image_shape: tuple,
        max_number_of_faces: int,
        train_path: str, 
        val_path: str,
        center_bbox: bool = True):
        '''
            The structure of each entery is:
                File name
                Number of faces
                [
                    [x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose]
                ]
            (the last line is repeated for the number of faces that exist in the image)
            the formatted output looks like this (example):
                {
                    'image_path': '.\\WIDER-Face\\train\\images\\0--Parade\\0_Parade_marchingband_1_849.jpg',
                    'number_of_faces': 1,
                    'faces': [
                        {
                            'bbox': [449, 330, 122, 149],
                            'indicators': [0, 0, 0, 0, 0, 0]
                        }
                    ]
                }
        '''
        super(type(self), self).__init__()

        self.output_image_shape = (output_image_shape[1], output_image_shape[0]) # H, W
        self.max_number_of_faces = max_number_of_faces
        self.center_bbox = center_bbox

        train_annotations = self.extract_formated_annotations(train_path, "train_bbx_gt.txt")
        val_annotations = self.extract_formated_annotations(val_path, "val_bbx_gt.txt")
        self.annotation_datapoints = [*train_annotations, *val_annotations]


    def __len__(self):
        return len(self.annotation_datapoints)

    def __getitem__(self, idx):

        annotation = self.annotation_datapoints[idx]

        image = read_image(annotation["imagePath"]).to(dtype=T.float32)

        bboxes = [self.format_bbox(annotation["faces"][i]["faceBbox"], image.shape[1:]) for i in range(min(annotation["numberOfFaces"], self.max_number_of_faces))]
        padding = [T.zeros(4, dtype=T.float32) for _ in range(self.max_number_of_faces - len(bboxes))]
        bboxes = T.cat([*bboxes, *padding]).reshape(-1, 4)

        image = resize(image, self.output_image_shape, antialias=True)

        datapoint = {
            "image": image,
            "faceBbox": bboxes
        }

        return datapoint

    def format_bbox(self, bbox: tuple, current_image_shape: tuple):
        return T.tensor(scale_bbox(bbox, current_image_shape[::-1], self.output_image_shape[::-1]), dtype=T.float32)


    def extract_formated_annotations(self, path: str, annotation_file_name: str):
        
        formated_datapoints = []

        with open(os.path.join(path, annotation_file_name), "r") as f:

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
                    
                    x1, y1, w, h = [int(x) for x in face[:4]]

                    if self.center_bbox:
                        x1 = x1 + (w / 2)
                        y1 = y1 + (h / 2)
                        w = w / 2
                        h = h / 2

                    formated_faces.append({
                        "faceBbox": [x1, y1, w, h],
                        "indicators": [int(x) for x in face[4:]]
                    })
                
                formated_datapoints.append({
                    "imagePath": os.path.join(path, "images", image_folder, image_name),
                    "numberOfFaces": number_of_faces,
                    "faces": formated_faces
                })
        return formated_datapoints

class WFLWDataset(data.Dataset):

    def __init__(
        self,
        output_full_image_shape: tuple,
        output_face_image_shape: tuple,
        image_path: str,
        annotation_path: str,
        center_bbox: bool = True):
        '''
        file structure from the README:
        
            coordinates of 98 landmarks (196) + 
            coordinates of upper left corner and lower right corner of detection rectangle (4) + 
            attributes annotations (6) + 
            image name (1)
            
            namely
            
            x0 y0 ... x97 y97 
            x_min_rect y_min_rect x_max_rect y_max_rect 
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
            image_name
        '''
        super(type(self), self).__init__()
        
        print("WARNING: WFLW dataset currently returns one image / face pair even when there are multiple faces in the image")
        
        self.center_bbox = center_bbox
        
        self.output_full_image_shape = (output_full_image_shape[1], output_full_image_shape[0])
        self.output_face_image_shape = (output_face_image_shape[1], output_face_image_shape[0])
        
        self.image_folder_path = image_path
        
        train_datapoints = self.extract_formatted_datapoints(os.path.join(annotation_path, "train.txt"))
        validation_datapoints = self.extract_formatted_datapoints(os.path.join(annotation_path, "validation.txt"))
        
        self.all_datapoints = [*train_datapoints, *validation_datapoints]
        
    def __len__(self):
        return (len(self.all_datapoints))
    
    def __getitem__(self, idx):
        
        datapoint = self.all_datapoints[idx]
        
        full_image = read_image(datapoint["imagePath"]).to(T.float32)
        keypoints = T.tensor(datapoint["keypoints"], dtype=T.float32)
        indicators = T.tensor(datapoint["indicators"], dtype=T.float32)
        bbox = datapoint["faceBbox"]
        bbox = [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
        
        # create face crop
        face_image = crop(full_image, bbox[1], bbox[0], bbox[3], bbox[2])
        
        # scale keypoints acording to the new face crop size
        # kp_scale_factor = T.tensor([face_image.shape[2], face_image.shape[1]])
        scaled_keypoints = (keypoints - T.tensor(bbox[:2])) / T.tensor(face_image.shape[::-1][:2]) * T.tensor(self.output_face_image_shape[::-1])
        
        # scale face crop
        face_image = resize(face_image, self.output_face_image_shape, antialias=True)
        
        # scale global keypoints acording to the new full image size
        keypoints = keypoints / T.tensor(full_image.shape[::-1][:2]) * T.tensor(self.output_full_image_shape[::-1]).float()
        
        # scale bounding boxes acording to the new full image size
        bbox = scale_bbox(bbox, full_image.shape[::-1], self.output_full_image_shape[::-1])
        
        # scale full image
        full_image = resize(full_image, self.output_full_image_shape, antialias=True)
        
        
        # create center bounding boxes
        if self.center_bbox:
            bbox = [bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2), bbox[2] / 2, bbox[3] / 2]
        
        return {
            "fullImage": full_image,
            "faceBbox": T.tensor(bbox,dtype=T.float32),
            "faceImage": face_image, # in format [Channels, Height, Width]
            "globalKeypoints": keypoints, # in format [Width, Height] in image size
            "localKeypoints": scaled_keypoints, # in format [Width, Height] in face crop size
            "indicators": indicators
        }

    def extract_formatted_datapoints(self, path):
        individual_datapoints = []
        with open(path) as f:
            for l in f.readlines():
                individual_datapoints.append(self.process_annotation(l))
        
        return individual_datapoints

    def process_annotation(self, line):
        line = line.split(" ")
        return {
            "imagePath": os.path.join(self.image_folder_path, line[-1])[:-1],
            "keypoints": np.array([float(a) for a in line[:196]]).reshape((98, 2)),
            "faceBbox": np.array([int(ell) for ell in line[196:200]], dtype=np.int16),
            "indicators": np.array([int(ell) for ell in line[200:206]])
        }

class COFWColorDataset(data.Dataset):
    def __init__(
        self, 
        output_full_image_shape: tuple, 
        output_face_image_shape: tuple,
        data_path: str,
        center_bbox: bool = True):
        
        super(type(self), self).__init__()
        
        self.center_bbox = center_bbox
        
        self.output_full_image_shape = [output_full_image_shape[1], output_full_image_shape[0]]
        self.output_face_image_shape = [output_face_image_shape[1], output_face_image_shape[0]]
        
        self.train_file, train_datapoints = self.extract_formatted_datapoints(os.path.join(data_path, "color_train.mat"), is_train=True)
        self.test_file, test_datapoints = self.extract_formatted_datapoints(os.path.join(data_path, "color_test.mat"), is_train=False)
        self.all_datapoints = [*train_datapoints, *test_datapoints]
        
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self, idx):
        
        is_train, image_ref, bbox, phis = self.all_datapoints[idx]
        
        bbox = [*map(int, bbox)]
        
        image = self.train_file[image_ref] if is_train else self.test_file[image_ref]
        image = T.tensor(np.array(image), dtype=T.float32).permute(0, 2, 1)
        
        keypoints = T.tensor(phis, dtype=T.float32)[:58].reshape(2, 29).permute(1, 0)
        occlusion = T.tensor(phis, dtype=T.float32)[58:]
        
        # crop face
        face_image = crop(image, bbox[1], bbox[0], bbox[3], bbox[2])
        
        # scale keypoints acording to the new face crop size
        scaled_keypoints = (keypoints - T.tensor(bbox[:2])) / T.tensor(face_image.shape[::-1][:2]) * T.tensor(self.output_face_image_shape[::-1])
        
        # scale face crop
        face_image = resize(face_image, self.output_face_image_shape, antialias=True)
        
        # scale global keypoints acording to the new full image size
        keypoints = keypoints / T.tensor(image.shape[::-1][:2]) * T.tensor(self.output_full_image_shape[::-1]).float()
        
        # scale bounding boxes acording to the new full image size
        bbox = scale_bbox(bbox, image.shape[::-1], self.output_full_image_shape[::-1])
        
        # center bounding boxes
        if self.center_bbox:
            bbox = [bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2), bbox[2] / 2, bbox[3] / 2]
        
        # scale full image
        image = resize(image, self.output_full_image_shape, antialias=True)
        
        return {
            "fullImage": image,
            "faceImage": face_image,
            "faceBbox": T.tensor(bbox, dtype=T.float32),
            "localKeypoints": scaled_keypoints,
            "globalKeypoints": keypoints,
            "keypointOcclusion": occlusion
        }
    
    def extract_formatted_datapoints(self, path: str, is_train: bool):
        
        file = h5py.File(path, "r")
        keys = list(file.get("/"))
        
        IsT = np.array(file.get(keys[1])).squeeze()
        bboxes = np.array(file.get(keys[2])).squeeze().T
        phis = np.array(file.get(keys[3])).squeeze().T
        
        return file, [(is_train, *p) for p in zip(IsT, bboxes, phis)]

class MPIIDataset(data.Dataset):
    
    def __init__(
        self,
        output_full_image_shape: tuple,
        output_person_image_shape: tuple,
        annotation_path: str,
        image_folder_path: str,
        center_bbox: bool = True):
        super(type(self), self).__init__()
        
        self.output_full_image_shape = [output_full_image_shape[1], output_full_image_shape[0]]
        self.output_person_image_shape = [output_person_image_shape[1], output_person_image_shape[0]]
        
        self.center_bbox = center_bbox
        
        self.image_folder_path = image_folder_path
        
        with open(os.path.join(annotation_path, "trainval.json")) as f:
            self.datapoints = json.load(f)
        
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        dp = self.datapoints[idx]
        
        image = read_image(os.path.join(self.image_folder_path, dp["image"]))
        keypoints = T.tensor(dp["joints"], dtype=T.float32)
        visibility = T.tensor(dp["joints_vis"], dtype=T.float32)
        scale = T.tensor(dp["scale"], dtype=T.float32)
        
        visible_keypoints = keypoints[visibility == 1]
        
        # calculate bounding box from visible keypoints
        bbox = T.stack([visible_keypoints.min(dim=0)[0], visible_keypoints.max(dim=0)[0]])
        bbox = [*map(int, T.cat([bbox[0], bbox[1] - bbox[0]]))]
        
        # crop image and scale
        person_image = crop(image, bbox[1], bbox[0], bbox[3], bbox[2])
        
        # scale keypoints to person image size
        scaled_keypoints = (keypoints - T.tensor(bbox[:2])) / T.tensor(person_image.shape[::-1][:2]) * T.tensor(self.output_person_image_shape[::-1]).float()
        
        # scale person image
        person_image = resize(person_image, self.output_person_image_shape).to(dtype=T.float32)
        
        # scale keypoints to new image size
        keypoints = keypoints / T.tensor(image.shape[::-1][:2]) * T.tensor(self.output_full_image_shape[::-1]).float()
        
        # scale bbox
        bbox = scale_bbox(bbox, image.shape[::-1][:2], self.output_full_image_shape)
        
        # center bbox
        if self.center_bbox:
            bbox = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2] / 2, bbox[3] / 2]
        
        # scale image
        image = resize(image, self.output_full_image_shape).to(dtype=T.float32)
        
        return {
            "fullImage": image,
            "personImage": person_image,
            "bbox": T.tensor(bbox, dtype=T.float32),
            "globalKeypoints": keypoints,
            "localKeypoints": scaled_keypoints,
            "keypointVisibility": visibility,
            "scale": scale,
        }






# TODO: Finish this, missing load
class COCODataset(data.Dataset):
    def __init__(self):
        super(type(self), self).__init__()

        self.anno_list = []

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, idx):

        coco = self.anno_list[idx]["coco"]
        halpe = self.anno_list[idx]["halpe"]

        coco_img = cv2.imread(coco["image_path"])
        halpe_img = cv2.imread(halpe[""])


        coco_ordered_kpts_loc = self.ordered_kpts_to_tensor(coco["loc"])
        coco_ordered_kpts_vis = self.ordered_kpts_to_tensor(coco["vis"])
        halpe_ordered_kpts_loc = self.ordered_kpts_to_tensor(halpe["loc"])
        halpe_ordered_kpts_vis = self.ordered_kpts_to_tensor(halpe["vis"])

        coco_ordered_bp_kpts = self.ordered_bp_kpts_to_tensor(coco["body_part_kpts"])
        halpe_ordered_bp_kpts = self.ordered_bp_kpts_to_tensor(halpe["body_part_kpts"])
        
        coco_ordered_bp_kpts_vis = self.ordered_bp_kpts_visibility_to_tensor(coco["body_part_kpts"], coco["face_valid"], coco["left_hand_valid"], coco["right_hand_valid"], coco["foot_valid"])
        halpe_ordered_bp_kpts_vis = self.ordered_bp_kpts_visibility_to_tensor(halpe["body_part_kpts"], True, True, True, True)

        y = {
            "coco": {
                "img": coco_img.to(self.device),
                "bbox": coco["annotation"]["bbox"].to(self.device),
                
                "body_parts_kpts_loc": coco_ordered_bp_kpts.to(self.device),
                "body_part_kpts_vis": coco_ordered_bp_kpts_vis.to(self.device),

                "ordered_kpts_loc": coco_ordered_kpts_loc.to(self.device),
                "ordered_kpts_vis": coco_ordered_kpts_vis.to(self.device),
            },
            "halpe": {
                "img": halpe["img"].to(self.device),
                "bbox": halpe["annotation"]["bbox"].to(self.device),
                
                "body_parts_kpts_loc": halpe_ordered_bp_kpts.to(self.device),
                "body_parts_kpts_vis": halpe_ordered_bp_kpts_vis.to(self.device),
                
                "ordered_kpts_loc": halpe_ordered_kpts_loc.to(self.device),
                "ordered_kpts_vis": halpe_ordered_kpts_vis.to(self.device),
            }
        }

        return y

    def ordered_bp_kpts_visibility_to_tensor(self, kpts, face_valid, left_hand_valid, right_hand_valid, foot_valid):
        return T.cat([
            [1, 0] if face_valid and T.min(kpts["head"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["left_shoulder"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["right_shoulder"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["left_elbow"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["right_elbow"]) != 0 else [0, 1],
            [1, 0] if left_hand_valid and T.min(kpts["left_hand"]) != 0 else [0, 1],
            [1, 0] if right_hand_valid and T.min(kpts["right_hand"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["left_hip"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["right_hip"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["left_knee"]) != 0 else [0, 1],
            [1, 0] if T.min(kpts["right_knee"]) != 0 else [0, 1],
            [1, 0] if foot_valid and T.min(kpts["left_foot"]) != 0 else [0, 1],
            [1, 0] if foot_valid and T.min(kpts["right_foot"]) != 0 else [0, 1],
        ], dtype=T.float32)

    def ordered_bp_kpts_to_tensor(self, kpts):
        return T.tensor(np.concatenate([
            kpts["head"],
            kpts["left_shoulder"],
            kpts["right_shoulder"],
            kpts["left_elbow"],
            kpts["right_elbow"],
            kpts["left_hand"],
            kpts["right_hand"],
            kpts["left_hip"],
            kpts["right_hip"],
            kpts["left_knee"],
            kpts["right_knee"],
            kpts["left_foot"],
            kpts["right_foot"],
        ]), dtype=T.float32)

    def ordered_kpts_to_tensor(self, kpts):
        return T.tensor(np.concatenate([
            kpts["body"]["nose"], # tpn_pose_loc[0]
            kpts["body"]["left_eye"], # tpn_pose_loc[1]
            kpts["body"]["right_eye"], # tpn_pose_loc[2]
            kpts["body"]["left_ear"], # tpn_pose_loc[3]
            kpts["body"]["right_ear"], # tpn_pose_loc[4]
            kpts["body"]["left_shoulder"], # tpn_pose_loc[5]
            kpts["body"]["right_shoulder"], # tpn_pose_loc[6]
            kpts["body"]["left_elbow"], # tpn_pose_loc[7]
            kpts["body"]["right_elbow"], # tpn_pose_loc[8]
            kpts["body"]["left_wrist"], # tpn_pose_loc[9]
            kpts["body"]["right_wrist"], # tpn_pose_loc[10]
            kpts["body"]["left_hip"], # tpn_pose_loc[11]
            kpts["body"]["right_hip"], # tpn_pose_loc[12]
            kpts["body"]["left_knee"], # tpn_pose_loc[13]
            kpts["body"]["right_knee"], # tpn_pose_loc[14]
            kpts["body"]["left_ankle"], # tpn_pose_loc[15]
            kpts["body"]["right_ankle"], # tpn_pose_loc[16]
            kpts["body"]["left_heel"], # tpn_pose_loc[17]
            kpts["body"]["right_heel"], # tpn_pose_loc[18]
            kpts["body"]["left_big_toe"], # tpn_pose_loc[19]
            kpts["body"]["right_big_toe"], # tpn_pose_loc[20]
            kpts["body"]["left_small_toe"], # tpn_pose_loc[21]
            kpts["body"]["right_small_toe"], # tpn_pose_loc[22]
            kpts["face"]["jawline"], # tpn_pose_loc[23:40]
            kpts["face"]["brows"]["right"], # tpn_pose_loc[40:45]
            kpts["face"]["brows"]["left"], # tpn_pose_loc[45:50]
            kpts["face"]["nose"]["bridge"], # tpn_pose_loc[50:54]
            kpts["face"]["nose"]["bottom"], # tpn_pose_loc[54:59]
            kpts["face"]["eyes"]["left"], # tpn_pose_loc[59:65]
            kpts["face"]["eyes"]["right"], # tpn_pose_loc[65:71]
            kpts["face"]["mouth"]["corner"]["left"], # tpn_pose_loc[71]
            kpts["face"]["mouth"]["corner"]["right"], # tpn_pose_loc[72]
            kpts["face"]["mouth"]["upper"]["top"], # tpn_pose_loc[73:78]
            kpts["face"]["mouth"]["upper"]["bottom"], # tpn_pose_loc[78:81]
            kpts["face"]["mouth"]["lower"]["top"], # tpn_pose_loc[81:84]
            kpts["face"]["mouth"]["lower"]["bottom"], # tpn_pose_loc[84:89]
            kpts["left_hand"]["thumb"], # tpn_pose_loc[89:94]
            kpts["left_hand"]["index"], # tpn_pose_loc[94:98]
            kpts["left_hand"]["middle"], # tpn_pose_loc[98:102]
            kpts["left_hand"]["ring"], # tpn_pose_loc[102:106]
            kpts["left_hand"]["pinky"], # tpn_pose_loc[106:109]
            kpts["right_hand"]["thumb"], # tpn_pose_loc[109:114]
            kpts["right_hand"]["index"], # tpn_pose_loc[114:118]
            kpts["right_hand"]["middle"], # tpn_pose_loc[118:122]
            kpts["right_hand"]["ring"], # tpn_pose_loc[122:126]
            kpts["right_hand"]["pinky"], # tpn_pose_loc[126:131]
        ]), dtype = T.float32)





# TODO: refactoring
class COCOWholeBody(data.Dataset):

    def __init__(self, annotation_json_path: str, image_folder_path: str):
        super(type(self), self).__init__()

        x = json.load(open(annotation_json_path, "r"))
        out = {}

        for img in x["images"]:
            out[img["id"]] = {
                "width": img["width"],
                "height": img["height"],
                "image_path": image_folder_path + img["file_name"],
                "annotation": []
            }

        for a in x["annotations"]:
            if a["image_id"] in out:

                o = {}
                
                o["segmentation"] = np.array(a["segmentation"])
                o["num_kpts"] = a["num_keypoints"]
                o["id"] = a["id"]
                o["area"] = a["area"]
                o["iscrowd"] = a["iscrowd"]

                # TODO: format category
                o["category_id"] = a["category_id"]

                o["bbox"] = a["bbox"]
                o["face_bbox"] = None if all([fb == 0. for fb in a["face_box"]]) else a["face_box"]
                o["left_hand_bbox"] = None if all([fb == 0. for fb in a["lefthand_box"]]) else a["lefthand_box"]
                o["right_hand_bbox"] = None if all([fb == 0. for fb in a["righthand_box"]]) else a["righthand_box"]

                o["face_valid"] = a["face_valid"]
                o["left_hand_valid"] = a["lefthand_valid"]
                o["right_hand_valid"] = a["righthand_valid"]
                o["foot_valid"] = a["foot_valid"]

                kpts = [*a["keypoints"], *a["foot_kpts"], *a["face_kpts"], *a["lefthand_kpts"], *a["righthand_kpts"]]
                kpts = np.array(kpts).reshape((133, 3))

                o["keypoints"] = {
                    "body": {
                        "nose": kpts[0],
                        "left_eye": kpts[1],
                        "right_eye": kpts[2],
                        "left_ear": kpts[3],
                        "right_ear": kpts[4],
                        "left_shoulder": kpts[5],
                        "right_shoulder": kpts[6],
                        "left_elbow": kpts[7],
                        "right_elbow": kpts[8],
                        "left_wrist": kpts[9],
                        "right_wrist": kpts[10],
                        "left_hip": kpts[11],
                        "right_hip": kpts[12],
                        "left_knee": kpts[13],
                        "right_knee": kpts[14],
                        "left_ankle": kpts[15],
                        "right_ankle": kpts[16],
                        "left_heel": kpts[19],
                        "right_heel": kpts[22],
                        "left_big_toe": kpts[17],
                        "right_big_toe": kpts[20],
                        "left_small_toe": kpts[18],
                        "right_small_toe": kpts[21],
                    },
                    "face": {
                        "jawline": kpts[23:40],
                        "brows": {
                            "right": kpts[40:45],
                            "left": kpts[45:50],
                        },
                        "nose": {
                            "bridge": kpts[50:54],
                            "bottom": kpts[54:59],
                        },
                        "eyes": {
                            "right": kpts[59:65],
                            "left": kpts[65:71],
                        },
                        "mouth": {
                            "corner": {
                                "right": kpts[71],
                                "left": kpts[77],
                            },
                            "upper": {
                                "top": kpts[72:77],
                                "bottom": kpts[88:91],
                            },
                            "lower": {
                                "top": kpts[84:87],
                                "bottom": kpts[78:83],
                            }
                        }
                    },
                    "left_hand": { 
                        "thumb": kpts[91:96],
                        "index": kpts[96:100],
                        "middle": kpts[100:104],
                        "ring": kpts[104:108],
                        "pinky": kpts[108:112],
                    },
                    "right_hand": {
                        "thumb": kpts[112:117],
                        "index": kpts[117:121],
                        "middle": kpts[121:125],
                        "ring": kpts[125:129],
                        "pinky": kpts[129:133],
                    }
                }

                out[a["image_id"]]["annotation"].append(o)

        out_ = {}
        for k in out:
            if len(out[k]["annotation"]) == 1:
                out_[k] = deepcopy(out[k])
                out_[k]["annotation"] = deepcopy(out[k]["annotation"][0])

                # BBOX
                x_min, y_min, w, h = [int(a) for a in out_[k]["annotation"]["bbox"]]
                x_max = x_min + w
                y_max = y_min + h
                out_[k]["annotation"]["bbox"] = np.array([x_min, y_min, x_max, y_max])

                out_[k]["body_part_kpts"] = {
                    "head": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["nose"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_eye"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_eye"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_ear"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_ear"]
                    ]), axis = 0),
                    "left_shoulder": out_[k]["annotation"]["keypoints"]["body"]["left_shoulder"],
                    "right_shoulder": out_[k]["annotation"]["keypoints"]["body"]["right_shoulder"],
                    "left_elbow": out_[k]["annotation"]["keypoints"]["body"]["left_elbow"],
                    "right_elbow": out_[k]["annotation"]["keypoints"]["body"]["right_elbow"],
                    "left_hip": out_[k]["annotation"]["keypoints"]["body"]["left_hip"],
                    "right_hip": out_[k]["annotation"]["keypoints"]["body"]["right_hip"],
                    "left_knee": out_[k]["annotation"]["keypoints"]["body"]["left_knee"],
                    "right_knee": out_[k]["annotation"]["keypoints"]["body"]["right_knee"],
                    "left_foot": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["left_ankle"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_heel"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_big_toe"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_small_toe"],
                    ]), axis = 0),
                    "right_foot": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["right_ankle"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_heel"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_big_toe"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_small_toe"],
                    ]), axis = 0),
                    "left_hand": T.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["left_wrist"],
                        out_[k]["annotation"]["keypoints"]["left_hand"]["thumb"],
                        out_[k]["annotation"]["keypoints"]["left_hand"]["index"],
                        out_[k]["annotation"]["keypoints"]["left_hand"]["middle"],
                        out_[k]["annotation"]["keypoints"]["left_hand"]["ring"],
                        out_[k]["annotation"]["keypoints"]["left_hand"]["pinky"],
                    ]), axis = 0),
                    "right_hand": T.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["right_wrist"],
                        out_[k]["annotation"]["keypoints"]["right_hand"]["thumb"],
                        out_[k]["annotation"]["keypoints"]["right_hand"]["index"],
                        out_[k]["annotation"]["keypoints"]["right_hand"]["middle"],
                        out_[k]["annotation"]["keypoints"]["right_hand"]["ring"],
                        out_[k]["annotation"]["keypoints"]["right_hand"]["pinky"],
                    ]), axis = 0),
                }

        del out
        return out_
    
# TODO: refactoring
class HalpeFullBody(data.Dataset):

    def __init__(self, annotation_json_path: str, image_folder_path: str):
        super(type(self), self).__init__()
        
        x = json.load(open(anno_path, "r"))
        out = {}

        for img in x["images"]:
            out[img["id"]] = {
                "width": img["width"],
                "height": img["height"],
                "image_path": img_folder + img["file_name"],
                "annotation": []
            }

        for a in x["annotations"]:
            if a["image_id"] in out:

                o = {}
                
                o["bbox"] = a["bbox"]

                kpts = a["keypoints"]
                kpts = np.array(kpts).reshape((136, 3))

                o["keypoints"] = {
                    "body": {
                        "nose": kpts[0],
                        "left_eye": kpts[1],
                        "right_eye": kpts[2],
                        "left_ear": kpts[3],
                        "right_ear": kpts[4],
                        "left_shoulder": kpts[5],
                        "right_shoulder": kpts[6],
                        "left_elbow": kpts[7],
                        "right_elbow": kpts[8],
                        "left_wrist": kpts[9],
                        "right_wrist": kpts[10],
                        "left_hip": kpts[11],
                        "right_hip": kpts[12],
                        "left_knee": kpts[13],
                        "right_knee": kpts[14],
                        "left_ankle": kpts[15],
                        "right_ankle": kpts[16],
                        # "head": kpts[17],
                        # "neck": kpts[18],
                        # "hip": kpts[19],
                        "left_big_toe": kpts[20],
                        "right_big_toe": kpts[21],
                        "left_small_toe": kpts[22],
                        "right_small_toe": kpts[23],
                        "left_heel": kpts[24],
                        "right_heel": kpts[25],
                    },
                    "face": { # 26-93
                        "jawline": kpts[26:43],
                        "brows": {
                            "right": kpts[43:48],
                            "left": kpts[48:53],
                        },
                        "nose": {
                            "bridge": kpts[53:57],
                            "bottom": kpts[57:62],
                        },
                        "eyes": {
                            "right": kpts[62:68],
                            "left": kpts[68:74],
                        },
                        "mouth": {
                            "corner": {
                                "right": kpts[74],
                                "left": kpts[80],
                            },
                            "upper": {
                                "top": kpts[75:80],
                                "bottom": kpts[87:90],
                            },
                            "lower": {
                                "top": kpts[91:94],
                                "bottom": kpts[81:86],
                            }
                        }
                    },
                    "left_hand": { # 94-114
                        "thumb": kpts[94:99],
                        "index": kpts[99:94+9],
                        "middle": kpts[94+9:94+13],
                        "ring": kpts[94+13:94+17],
                        "pinky": kpts[94+17:94+21]
                    },
                    "right_hand": { # 115-136
                        "thumb": kpts[115:120],
                        "index": kpts[120:115+9],
                        "middle": kpts[115+9:115+13],
                        "ring": kpts[115+13:115+17],
                        "pinky": kpts[115+17:115+21],
                    }
                }



                out[a["image_id"]]["annotation"].append(o)


        out_ = {}
        for k in out:
            if len(out[k]["annotation"]) == 1:
                out_[k] = deepcopy(out[k])
                out_[k]["annotation"] = deepcopy(out[k]["annotation"][0])

                # BBOX
                x_min, y_min, w, h = [int(a) for a in out_[k]["annotation"]["bbox"]]
                x_max = x_min + w
                y_max = y_min + h
                out_[k]["annotation"]["bbox"] = np.array([x_min, y_min, x_max, y_max])

                out_[k]["body_part_kpts"] = {
                    "head": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["nose"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_eye"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_eye"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_ear"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_ear"]
                    ]), axis = 0),
                    "left_shoulder": out_[k]["annotation"]["keypoints"]["body"]["left_shoulder"],
                    "right_shoulder": out_[k]["annotation"]["keypoints"]["body"]["right_shoulder"],
                    "left_elbow": out_[k]["annotation"]["keypoints"]["body"]["left_elbow"],
                    "right_elbow": out_[k]["annotation"]["keypoints"]["body"]["right_elbow"],
                    "left_hip": out_[k]["annotation"]["keypoints"]["body"]["left_hip"],
                    "right_hip": out_[k]["annotation"]["keypoints"]["body"]["right_hip"],
                    "left_knee": out_[k]["annotation"]["keypoints"]["body"]["left_knee"],
                    "right_knee": out_[k]["annotation"]["keypoints"]["body"]["right_knee"],
                    "left_foot": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["left_ankle"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_heel"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_big_toe"],
                        out_[k]["annotation"]["keypoints"]["body"]["left_small_toe"],
                    ]), axis = 0),
                    "right_foot": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["right_ankle"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_heel"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_big_toe"],
                        out_[k]["annotation"]["keypoints"]["body"]["right_small_toe"],
                    ]), axis = 0),
                    "left_hand": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["left_wrist"],
                        *out_[k]["annotation"]["keypoints"]["left_hand"]["thumb"],
                        *out_[k]["annotation"]["keypoints"]["left_hand"]["index"],
                        *out_[k]["annotation"]["keypoints"]["left_hand"]["middle"],
                        *out_[k]["annotation"]["keypoints"]["left_hand"]["ring"],
                        *out_[k]["annotation"]["keypoints"]["left_hand"]["pinky"],
                    ]), axis = 0),
                    "right_hand": np.mean(np.stack([
                        out_[k]["annotation"]["keypoints"]["body"]["right_wrist"],
                        *out_[k]["annotation"]["keypoints"]["right_hand"]["thumb"],
                        *out_[k]["annotation"]["keypoints"]["right_hand"]["index"],
                        *out_[k]["annotation"]["keypoints"]["right_hand"]["middle"],
                        *out_[k]["annotation"]["keypoints"]["right_hand"]["ring"],
                        *out_[k]["annotation"]["keypoints"]["right_hand"]["pinky"],
                    ]), axis = 0),
                }

        del out
        return out_

# TODO
class DexYCB(data.Dataset):
    
    def __init__(self, annotation_json_path: str, image_folder_path: str):
        super(type(self), self).__init__()

# TODO
class FreiHAND(data.Dataset):
    
    def __init__(self, annotation_json_path: str, image_folder_path: str):
        super(type(self), self).__init__()

# TODO
class ICVL(data.Dataset):
    def __init__(self, annotation_json_path: str, image_folder_path: str):
        super(type(self), self).__init__()
