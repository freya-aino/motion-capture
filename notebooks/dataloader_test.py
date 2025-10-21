import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import json
    import os
    import re
    import pandas as pd
    import numpy as np
    import torch as T

    from copy import deepcopy
    from torch.utils import data
    from torchvision.transforms.functional import resize, crop
    from torchvision.io import read_image, ImageReadMode
    from torch.nn.functional import one_hot, pad

    from motion_capture.data.datasets import scale_points
    return T, data, json, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# CelebA""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from motion_capture.data.datasets import CelebA
    image_shape = (448, 224)
    celeba_dataset = CelebA(annotatin_path='\\\\192.168.2.206\\data\\datasets\\CelebA\\Anno', image_path='\\\\192.168.2.206\\data\\datasets\\CelebA\\img\\img_align_celeba\\img_celeba', image_shape_WH=image_shape)
    return celeba_dataset, image_shape, patches, plt


@app.cell
def _(T, celeba_dataset, image_shape, patches, plt):
    _test_i = 10
    _x, _y = celeba_dataset[_test_i]
    plt.imshow((_x * 255).permute(1, 2, 0).round().byte().numpy())
    _x1, _y1, _x2, _y2 = _y['bbox'] * T.tensor(image_shape * 2)
    plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='r', facecolor='none'))
    for _kpt in _y['keypoints']:
        plt.scatter(_kpt[0] * image_shape[0], _kpt[1] * image_shape[1], c='r')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# FreiHAND""")
    return


@app.cell
def _(data, json, os):
    class FreiHAND(data.Dataset):

        def __init__(self, path: str):
            super().__init__()
            train_path = os.path.join(path, 'FreiHAND_pub_v2')
            self.rgb_image_path = os.path.join(train_path, 'training', 'rgb')
            self.mask_image_path = os.path.join(train_path, 'training', 'mask')
            with open(os.path.join(train_path, 'training_scale.json')) as f:
                self.scale = json.load(f)
            with open(os.path.join(train_path, 'training_mano.json')) as f:
                self.mano = json.load(f)
            with open(os.path.join(train_path, 'training_xyz.json')) as f:
                self.xyz = json.load(f)  # val_path = os.path.join(path, "FreiHAND_pub_v2_eval")
    freihand_dataset = FreiHAND(path='\\\\192.168.2.206\\data\\datasets\\FreiHAND')
    return (freihand_dataset,)


@app.cell
def _(freihand_dataset):
    freihand_dataset.mano[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# WIDER Face""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import WIDERFace
    image_shape_1 = (448, 224)
    wider_face_dataset = WIDERFace(path='//192.168.2.206/data/datasets/WIDER-Face', image_shape_WH=image_shape_1, max_number_of_faces=10)
    len(wider_face_dataset)
    return image_shape_1, wider_face_dataset


@app.cell
def _(wider_face_dataset):
    import pickle

    pickle.dumps(wider_face_dataset.all_datapoints)
    return


@app.cell
def _(T, image_shape_1, patches, plt, wider_face_dataset):
    _test_i = 3
    _x, _y = wider_face_dataset[_test_i]
    _x = (_x.permute(1, 2, 0) * 255).round().byte().numpy()
    plt.imshow(_x)
    for _i, _bbox in enumerate(_y['bboxes']):
        if not _y['validity'].argmax(1)[_i]:
            continue
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_1 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='red', facecolor='none'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# WFLW Face Recognition""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import WFLW
    image_shape_2 = (448, 224)
    wflw_dataset = WFLW(image_shape_WH=image_shape_2, path='//192.168.2.206/data/datasets/WFLW', max_number_of_faces=1)
    len(wflw_dataset)
    return image_shape_2, wflw_dataset


@app.cell
def _(T, image_shape_2, patches, plt, wflw_dataset):
    _test_i = 10
    _x, _y = wflw_dataset[_test_i]
    _x = (_x.permute(1, 2, 0) * 255).round().byte().numpy()
    plt.imshow(_x)
    for _i, _bbox in enumerate(_y['bboxes']):
        if _y['validity'].argmax(1)[_i] == 0:
            continue
        print(_bbox * T.tensor(image_shape_2 * 2))
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_2 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='red', facecolor='none'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# COFW Face Detection""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import COFWFaceDetection
    image_shape_3 = (224, 224)
    cofw_color_dataset = COFWFaceDetection(path='//192.168.2.206/data/datasets/COFW/', image_shape_WH=image_shape_3)
    len(cofw_color_dataset)
    return cofw_color_dataset, image_shape_3


@app.cell
def _(T, cofw_color_dataset, image_shape_3, patches, plt):
    _test_i = 115
    _x, _y = cofw_color_dataset[_test_i]
    _x = (_x.permute(1, 2, 0) * 255).round().byte().numpy()
    plt.imshow(_x)
    _x1, _y1, _x2, _y2 = _y['bbox'] * T.tensor(image_shape_3 * 2)
    plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='red', facecolor='none'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# COCO Global Person Instance Segmentation""")
    return


@app.cell
def _():
    # class COCO2017GlobalPersonInstanceSegmentation(data.Dataset):

    #     def __init__(
    #         self,
    #         image_folder_path: str,
    #         annotation_folder_path: str,
    #         image_shape_WH: tuple,
    #         max_num_persons: int,
    #         max_segmentation_points: int = 100,
    #         min_bbox_size: int = 50):

    #         super().__init__()

    #         self.max_num_persons = max_num_persons
    #         self.max_segmentation_points = max_segmentation_points
    #         self.image_shape = image_shape_WH

    #         with open(os.path.join(annotation_folder_path, "person_keypoints_train2017.json"), "r") as f:
    #             train_json = json.load(f)
    #         with open(os.path.join(annotation_folder_path, "person_keypoints_val2017.json"), "r") as f:
    #             val_json = json.load(f)

    #         images = pd.DataFrame.from_records((*train_json["images"], *val_json["images"]))
    #         annotations = pd.DataFrame.from_records((*train_json["annotations"], *val_json["annotations"]))

    #         self.all_datapoints = pd.merge(
    #             left = annotations,
    #             right = images,
    #             left_on = "image_id",
    #             right_on = "id",
    #         )

    #         self.all_datapoints["image_path"] = image_folder_path + "/" + self.all_datapoints["file_name"]
    #         self.all_datapoints = self.all_datapoints[self.all_datapoints["bbox"].apply(lambda x: x[2] * x[3] > (min_bbox_size**2))]
    #         self.all_datapoints = self.all_datapoints[self.all_datapoints["segmentation"].apply(type) == list]

    #         self.all_datapoints.drop(columns=[
    #             "num_keypoints", "area", "iscrowd", "keypoints",
    #             "image_id", "category_id", "id_x", "license", "file_name",
    #             "coco_url", "height", "width", "date_captured", "flickr_url", "id_y"
    #             ], inplace=True)

    #         self.all_datapoints = self.all_datapoints.groupby("image_path")
    #         person_count_mask = self.all_datapoints.size() <= max_num_persons
    #         self.all_datapoints = self.all_datapoints.aggregate(lambda x: x.tolist())
    #         self.all_datapoints = self.all_datapoints[person_count_mask]
    #         self.all_datapoints.reset_index(inplace=True)

    #     def __len__(self):
    #         return len(self.all_datapoints)

    #     def __getitem__(self, idx):
    #         dp = self.all_datapoints.iloc[idx]

    #         image = read_image(dp["image_path"], mode=ImageReadMode.RGB).to(dtype=T.float32)
    #         bboxes = dp["bbox"]
    #         segmentations = dp["segmentation"]

    #         bboxes_ = T.zeros(self.max_num_persons, 4)

    #         bboxes = T.tensor(bboxes, dtype=T.float32).reshape(-1, 2, 2) # ([scale_points(T.tensor(bbox).reshape(2, 2), self.image_shape[::-1], [1, 1]) for bbox in bboxes])
    #         bboxes = scale_points(bboxes, image.shape[::-1][:2], [1, 1])
    #         bboxes[:, 1, :] += bboxes[:, 0, :]
    #         bboxes = bboxes.reshape(-1, 4)

    #         bboxes_[:bboxes.shape[0]] = bboxes[:]


    #         segmentations_ = T.zeros(self.max_num_persons, self.max_segmentation_points, 2)
    #         for seg in segmentations:
    #             seg = T.tensor(seg, dtype=T.float32).reshape(-1, 2)[:self.max_segmentation_points]
    #             seg = scale_points(seg, image.shape[::-1][:2], [1, 1])
    #             seg = pad(seg, (0, 0, 0, max(0, self.max_segmentation_points - seg.shape[0])), value=0)
    #             segmentations_[:seg.shape[0]] = seg[:]

    #         # # validity mask
    #         # bbox_validity_mask = T.zeros(self.max_num_persons).bool()
    #         # bbox_validity_mask[(bboxes != 0).all(-1)] = True
    #         # segmentation_validity_mask = T.zeros(self.max_num_persons, self.max_segmentation_points).bool()
    #         # segmentation_validity_mask[(segmentations != 0).all(-1)] = True

    #         # resize full image
    #         image = resize(image, self.image_shape[::-1]) / 255

    #         # return concatenation of all datapoints
    #         return image, {
    #             "bboxes": bboxes_,
    #             "bboxValidity": 0,
    #             "segmentations": segmentations_,
    #             "segmentationValidity": 0
    #         }
    return


@app.cell
def _(COCO2017GlobalPersonInstanceSegmentation):
    image_shape_4 = (448, 224)
    person_instance_dataset = COCO2017GlobalPersonInstanceSegmentation(image_folder_path='//192.168.2.206/data/datasets/COCO2017/images', annotation_folder_path='//192.168.2.206/data/datasets/COCO2017/annotations', image_shape_WH=image_shape_4, max_num_persons=1, max_segmentation_points=100)
    # from motion_capture.data.datasets import COCO2017GlobalPersonInstanceSegmentation
    len(person_instance_dataset)
    return image_shape_4, person_instance_dataset


@app.cell
def _(T, image_shape_4, patches, person_instance_dataset, plt):
    _test_i = 79
    _x, _y = person_instance_dataset[_test_i]
    plt.imshow((_x.permute(1, 2, 0) * 255).round().byte().numpy())
    for _i, _bbox in enumerate(_y['bboxes']):
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_4 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='red', facecolor='none'))
    for _i, segmentation in enumerate(_y['segmentations']):
        for _j, point in enumerate(segmentation):
            _x_, _y_ = point * T.tensor(image_shape_4)
            plt.plot(_x_, _y_, color='red', marker='.')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# COCO Person Keypoints""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import COCO2017PersonKeypointsDataset
    image_shape_5 = (448, 224)
    person_keypoints_dataset = COCO2017PersonKeypointsDataset(image_folder_path='//192.168.2.206/data/datasets/COCO2017/images', annotation_folder_path='//192.168.2.206/data/datasets/COCO2017/annotations', image_shape_WH=image_shape_5, min_person_bbox_size=100)
    len(person_keypoints_dataset)
    return (
        COCO2017PersonKeypointsDataset,
        image_shape_5,
        person_keypoints_dataset,
    )


@app.cell
def _(T, image_shape_5, person_keypoints_dataset, plt):
    _test_i = 6
    _x, _y = person_keypoints_dataset[_test_i]
    plt.imshow((_x * 255).permute(1, 2, 0).round().byte().numpy())
    for _i, _kpt in enumerate(_y['keypoints']):
        if not _y['keypointValidity'].argmax(1)[_i]:
            continue
        _x_, _y_ = _kpt * T.tensor(image_shape_5)
        plt.plot(_x_, _y_, color='red', marker='.')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# HAKE Large""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import HAKELarge
    image_shape_6 = (448, 224)
    hake_dataset = HAKELarge(annotation_path='\\\\192.168.2.206\\data\\datasets\\HAKE\\Annotations', image_path='\\\\192.168.2.206\\data\\datasets\\HAKE-large', image_shape_WH=image_shape_6)
    len(hake_dataset)
    return hake_dataset, image_shape_6


@app.cell
def _(T, hake_dataset, image_shape_6, patches, plt):
    _test_i = 0
    _x, _y, z = hake_dataset[_test_i]
    plt.imshow((_x * 255).permute(1, 2, 0).byte().numpy())
    print(len(z['humanBboxes']))
    for _bbox in z['humanBboxes']:
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_6 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='red', facecolor='none'))
    for _bbox in z['objectBboxes']:
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_6 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='blue', facecolor='none'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MPII""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import MPIIDataset

    mpii_dataset = MPIIDataset(
        output_full_image_shape_WH=(448, 224),
        output_person_image_shape_WH=(224, 112),
        annotation_path="//192.168.2.206/data/datasets/MPII/annotations",
        image_folder_path="//192.168.2.206/data/datasets/MPII/images"
    )
    len(mpii_dataset)
    return (mpii_dataset,)


@app.cell
def _(mpii_dataset, patches, plt):
    _test_i = 110
    _, _ax = plt.subplots(1, 2)
    _ax[0].imshow(mpii_dataset[_test_i]['fullImage'].permute(1, 2, 0).byte().numpy())
    for dp in mpii_dataset[_test_i]['globalKeypoints']:
        _ax[0].plot(dp[0], dp[1], 'ro', markersize=2)
    # plot full image, keypoints, center and bounding box
    _x, _y = mpii_dataset[_test_i]['personBbox'][0]
    _w, _h = mpii_dataset[_test_i]['personBbox'][1]
    _ax[0].add_patch(patches.Rectangle((_x - _w, _y - _h), _w * 2, _h * 2, linewidth=1, edgecolor='red', facecolor='none'))
    _ax[1].imshow(mpii_dataset[_test_i]['personImage'].permute(1, 2, 0).byte().numpy())
    for dp in mpii_dataset[_test_i]['localKeypoints']:
        _ax[1].plot(dp[0], dp[1], 'ro', markersize=2)
    # plot person image and keypoints
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# COCO Panoptics Object Detection""")
    return


@app.cell
def _():
    from motion_capture.data.datasets import COCOPanopticsObjectDetection
    image_shape_7 = (448, 224)
    coco_panoptic_dataset = COCOPanopticsObjectDetection(image_folder_path='//192.168.2.206/data/datasets/COCO2017/images', panoptics_path='//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations', image_shape_WH=image_shape_7, max_number_of_instances=100)
    len(coco_panoptic_dataset)
    return coco_panoptic_dataset, image_shape_7


@app.cell
def _(T, coco_panoptic_dataset, image_shape_7, patches, plt):
    _test_i = 12
    _x, _y = coco_panoptic_dataset[_test_i]
    plt.imshow((_x * 255).permute(1, 2, 0).round().byte().numpy())
    for _i, _bbox in enumerate(_y['bboxes']):
        if not _y['validity'].argmax(1)[_i]:
            continue
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_7 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=1, edgecolor='red', facecolor='none'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# COCO Captions""")
    return


@app.cell
def _(data, json, os):
    class COCO2017CaptionsDataset(data.Dataset):

        def __init__(
            self,
            image_folder_path: str,
            annotation_folder_path: str):

            super().__init__()

            self.image_folder_path = image_folder_path
            self.annotation_path = annotation_folder_path

            with open(os.path.join(annotation_folder_path, "captions_train2017.json"), "r") as f:
                self.train_datapoints = json.load(f)

            with open(os.path.join(annotation_folder_path, "captions_val2017.json"), "r") as f:
                self.val_datapoints = json.load(f)

        def __len__(self):
            return len(self.all_datapoints)

        def __getitem__(self, idx):
            return self.all_datapoints[idx]
    return


@app.cell
def _(COCO2017PersonKeypointsDataset):
    coco_dataset = COCO2017PersonKeypointsDataset(image_folder_path='//192.168.2.206/data/datasets/COCO2017/images', annotation_folder_path='//192.168.2.206/data/datasets/COCO2017/annotations', output_full_image_shape_WH=(448, 224), output_person_image_shape_WH=(224, 112), load_val_only=False)
    len(coco_dataset)
    return (coco_dataset,)


@app.cell
def _(coco_dataset, patches, plt):
    _test_i = 20000
    full_image = coco_dataset[_test_i]['fullImage'].permute(1, 2, 0).round().byte().numpy()
    num_persons = coco_dataset[_test_i]['personImages'].shape[0]
    print(num_persons)
    fig, _ax = plt.subplots(num_persons + 1)
    _ax[0].imshow(full_image)
    for _i in range(num_persons):
        _kpts = coco_dataset[_test_i]['globalKeypoints'][_i]
    # create plot scale it to number of persons
        _vis = coco_dataset[_test_i]['keypointVisibility'][_i]
    # fig.set_size_inches(10, 10 * (num_persons + 1))
        _val = coco_dataset[_test_i]['keypointValidity'][_i]
        for _j in range(_kpts.shape[0]):
            if _vis[_j]:
                _ax[0].plot(_kpts[_j][0], _kpts[_j][1], 'bo', markersize=2)
            elif _val[_j]:
                _ax[0].plot(_kpts[_j][0], _kpts[_j][1], 'ro', markersize=2)
        _x, _y = coco_dataset[_test_i]['personBboxes'][_i][0]  # keypoints, bounding box and segmentation in full image
        _w, _h = coco_dataset[_test_i]['personBboxes'][_i][1]
        _ax[0].add_patch(patches.Rectangle((_x - _w, _y - _h), _w * 2, _h * 2, linewidth=1, edgecolor='red', facecolor='none'))
        person_image = coco_dataset[_test_i]['personImages'][_i].permute(1, 2, 0).round().byte().numpy()
        _ax[_i + 1].imshow(person_image)
        _kpts = coco_dataset[_test_i]['localKeypoints'][_i]
        _vis = coco_dataset[_test_i]['keypointVisibility'][_i]
        _val = coco_dataset[_test_i]['keypointValidity'][_i]
        for _j in range(_kpts.shape[0]):
            if _vis[_j]:
                _ax[_i + 1].plot(_kpts[_j][0], _kpts[_j][1], 'bo', markersize=2)
            elif _val[_j]:
                _ax[_i + 1].plot(_kpts[_j][0], _kpts[_j][1], 'ro', markersize=2)
    plt.show()  # plot individual person image and keypoints
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# COCO Person Whole Body""")
    return


@app.cell
def _():
    # class COCO2017PersonWholeBody(data.Dataset):

    #     def __init__(self, annotations_folder_path: str, image_folder_path: str, image_shape_WH: tuple, min_person_bbox_size: int = 100, padding: int = 20):
    #         super().__init__()

    #         self.annotations_folder_path = annotations_folder_path
    #         self.image_folder_path = image_folder_path
    #         self.image_shape = image_shape_WH
    #         self.padding = padding

    #         area = min_person_bbox_size ** 2

    #         with open(os.path.join(annotations_folder_path, "coco_wholebody_val_v1.0.json"), "r") as f:
    #             val_json = json.load(f)
    #         with open(os.path.join(annotations_folder_path, "coco_wholebody_train_v1.0.json"), "r") as f:
    #             train_json = json.load(f)

    #         images = pd.DataFrame.from_records((*train_json["images"], *val_json["images"]))
    #         annotations = pd.DataFrame.from_records((*train_json["annotations"], *val_json["annotations"]))
    #         # images = pd.DataFrame.from_records(val_json["images"])
    #         # annotations = pd.DataFrame.from_records(val_json["annotations"])

    #         self.all_datapoints = pd.merge(annotations, images, right_on="id", left_on="image_id")
    #         self.all_datapoints["image_path"] = self.image_folder_path + "/" + self.all_datapoints["file_name"]
    #         self.all_datapoints = self.all_datapoints[self.all_datapoints["bbox"].map(lambda x: x[2] * x[3] > area)]

    #         validity_mask = (self.all_datapoints["num_keypoints"] != 0) | self.all_datapoints["face_valid"] | self.all_datapoints["lefthand_valid"] | self.all_datapoints["righthand_valid"] | self.all_datapoints["foot_valid"]
    #         self.all_datapoints = self.all_datapoints[validity_mask]

    #         self.all_datapoints.reset_index(drop=True, inplace=True)

    #     def format_keypoints(self, datapoint):
    #         kpts = T.cat([
    #             T.tensor(datapoint["keypoints"]).reshape(-1, 3),
    #             T.tensor(datapoint["face_kpts"]).reshape(-1, 3),
    #             T.tensor(datapoint["lefthand_kpts"]).reshape(-1, 3),
    #             T.tensor(datapoint["righthand_kpts"]).reshape(-1, 3),
    #             T.tensor(datapoint["foot_kpts"]).reshape(-1, 3)
    #         ]).to(dtype=T.float32)

    #         kpts_visibility = kpts[:, 2] == 2
    #         kpts_validity = kpts[:, 2] > 0
    #         kpts = kpts[:, :2]

    #         return kpts, kpts_validity, kpts_visibility

    #     def __len__(self):
    #         return len(self.all_datapoints)

    #     def __getitem__(self, idx):

    #         datapoint = self.all_datapoints.iloc[idx]

    #         # load image
    #         image = read_image(datapoint["image_path"], mode=ImageReadMode.RGB).to(dtype=T.float32)
    #         person_bbox = T.tensor(datapoint["bbox"], dtype=T.int16).reshape(2, 2)
    #         face_bbox = T.tensor(datapoint["face_box"], dtype=T.int16).reshape(2, 2)
    #         lefthand_bbox =T.tensor(datapoint["lefthand_box"], dtype=T.int16).reshape(2, 2)
    #         righthand_bbox = T.tensor(datapoint["righthand_box"], dtype=T.int16).reshape(2, 2)
    #         all_keypoints, kpt_val, kpt_vis = self.format_keypoints(datapoint)

    #         # add padding to bounding boxes
    #         person_bbox[0] -= self.padding
    #         person_bbox[1] += self.padding * 2

    #         # crop persons
    #         person_crop = crop(image, person_bbox[0][1], person_bbox[0][0], person_bbox[1][1], person_bbox[1][0])
    #         person_crop = resize(person_crop, self.image_shape[::-1]) / 255

    #         # scale keypoints
    #         all_keypoints = scale_points(all_keypoints - person_bbox[0], person_bbox[1], [1, 1])

    #         # scale bounding boxes
    #         face_bbox = scale_points(face_bbox.to(dtype=T.float32) - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
    #         lefthand_bbox = scale_points(lefthand_bbox.to(dtype=T.float32) - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])
    #         righthand_bbox = scale_points(righthand_bbox.to(dtype=T.float32) - T.stack([person_bbox[0], T.zeros(2)]), person_bbox[1], [1, 1])

    #         # format bounding boxes
    #         person_bbox[1] += person_bbox[0]
    #         face_bbox[1] += face_bbox[0]
    #         lefthand_bbox[1] += lefthand_bbox[0]
    #         righthand_bbox[1] += righthand_bbox[0]

    #         # onehot encode keypoints
    #         kpt_val = one_hot(kpt_val.to(dtype=T.int64), 2)
    #         kpt_vis = one_hot(kpt_vis.to(dtype=T.int64), 2)

    #         return person_crop, {
    #             "keypoints": all_keypoints,
    #             "keypointsValidity": kpt_val,
    #             "keypointsVisibility": kpt_vis,
    #             "faceBbox": face_bbox.flatten(),
    #             "lefthandBbox": lefthand_bbox.flatten(),
    #             "righthandBbox": righthand_bbox.flatten(),
    #         }

    #     # TODO: create a function to formal keypoints back to readable
    #     def concat_keypoints(
    #         self,
    #         body_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints, foot_keypoints,
    #         body_visibility, face_visibility, left_hand_visibility, right_hand_visibility, foot_visibility,
    #         body_validity, face_validity, left_hand_validity, right_hand_validity, foot_validity):
    #         return (
    #             T.cat([body_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints, foot_keypoints]),
    #             T.cat([body_visibility, face_visibility, left_hand_visibility, right_hand_visibility, foot_visibility]),
    #             T.cat([body_validity, face_validity, left_hand_validity, right_hand_validity, foot_validity])
    #         )
    return


@app.cell
def _():
    from motion_capture.data.datasets import COCO2017PersonWholeBody
    image_shape_8 = (448, 224)
    coco_wholebody_dataset = COCO2017PersonWholeBody(annotations_folder_path='//192.168.2.206/data/datasets/COCO2017/annotations', image_folder_path='//192.168.2.206/data/datasets/COCO2017/images', image_shape_WH=image_shape_8, min_person_bbox_size=100)
    len(coco_wholebody_dataset)
    return coco_wholebody_dataset, image_shape_8


@app.cell
def _(T, coco_wholebody_dataset, image_shape_8, patches, plt):
    _test_i = 119
    _x, _y = coco_wholebody_dataset[_test_i]
    plt.imshow((_x * 255).permute(1, 2, 0).round().byte().numpy())
    _kpts = _y['keypoints']
    _vis = _y['keypointsVisibility']
    _val = _y['keypointsValidity']
    for _j in range(_kpts.shape[0]):
        if _vis.argmax(1)[_j] == 1:
            plt.plot(_kpts[_j][0] * image_shape_8[0], _kpts[_j][1] * image_shape_8[1], 'bo', markersize=5)
        elif _val.argmax(1)[_j] == 1:
            plt.plot(_kpts[_j][0] * image_shape_8[0], _kpts[_j][1] * image_shape_8[1], 'ro', markersize=5)
    bboxes = [_y['faceBbox'], _y['lefthandBbox'], _y['righthandBbox']]
    for _bbox in bboxes:
        _x1, _y1, _x2, _y2 = _bbox * T.tensor(image_shape_8 * 2)
        plt.gca().add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, linewidth=3, edgecolor='red', facecolor='none'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# hico_det""")
    return


@app.cell
def _():
    # hd = scipy.io.loadmat("./../_data/_usefull/_images/Halpe-FullBody/hico_det/anno.mat")
    # hd_bb = scipy.io.loadmat("./../_data/_usefull/_images/Halpe-FullBody/hico_det/anno_bbox.mat")

    # """

    # len(hd_bb["bbox_train"][0][i][1][0]) = 1

    # len(hd_bb["bbox_train"][0][i][0]) = 1
    # (name)


    # """
    return


@app.cell
def _():
    # hd_bb.keys()
    return


@app.cell
def _():
    # print(len(hd["list_train"]), len(hd["list_test"]))

    # hd_bb["bbox_train"][0][0][0]
    # hd_bb["bbox_train"][0][0][1]

    # for i in range(len(hd_bb["bbox_train"][0])):
    #     image_name = hd_bb["bbox_train"][0][i][0][0]

    #     width, height, depth = hd_bb["bbox_train"][0][i][1][0][0]
    #     width, height, depth = width[0][0], height[0][0], depth[0][0]


    # sum([len(hd_bb["bbox_train"][0][i][2][0]) for i in range(len(hd_bb["bbox_train"][0]))])

    # [b[1] for b in hd_bb["bbox_train"][0][0][2][0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# RHDv2""")
    return


@app.cell
def _():
    # arr = pickle.load(open("./../_data/_usefull/_images/RHD_published_v2/training/anno_training.pickle", "br"))

    """

    arr[i]["xyz"] = xyz keypoints
    arr[i]["uv_vis] = uv + visibility
    arr[i]["K"] = camera
    (i == image)


    """
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# HAKE""")
    return


@app.cell
def _():
    # hake_large_annotation = json.load(open("./../_data/_usefull/_images/HAKE/Annotations/hake_large_annotation.json", "r"))
    # hico_det_training_set_instance_level = json.load(open("./../_data/_usefull/_images/HAKE/Annotations/hico-det-training-set-instance-level.json", "r"))
    # hico_training_set_image_level = json.load(open("./../_data/_usefull/_images/HAKE/Annotations/hico-training-set-image-level.json", "r"))

    """

    hico training set image level

    {'arm_list': [0, 0, 0, 0, 1], 
    'foot_list': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'hand_list': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'head_list': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    'hip_list': [1, 0, 0, 0, 0], 
    'hoi_id': [153, 154, 155, 156], 
    'leg_list': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
    'parts_list': [1, 1, 1, 1, 1, 0, 1, 0, 0, 1]}


    hico det training set instance level

    {'dataset': 'hico-det', 
    'labels': [
        {
            'action_labels': 
                [{'human_part': 6, 'partstate': 0}, 
                {'human_part': 9, 'partstate': 0}, 
                {'human_part': 4, 'partstate': 0}, 
                {'human_part': 0, 'partstate': 0}, 
                {'human_part': 3, 'partstate': 0}], 
            'height': 480, 
            'hoi_id': 153, 
            'human_bbox': [208, 33, 427, 300], 
            'object_bbox': [59, 98, 572, 405], 
            'width': 640
        }, 
        {
            'action_labels': 
                [{'human_part': 4, 'partstate': 0}, 
                {'human_part': 1, 'partstate': 6}, 
                {'human_part': 2, 'partstate': 6}], 
            'height': 480, 
            'hoi_id': 156, 
            'human_bbox': [209, 26, 444, 317], 
            'object_bbox': [59, 99, 579, 395], 
            'width': 640
        },
            ...], 
    'path_prefix': 'hico_20160224_det/images/train2015'}



    hake large annotation

    {'dataset': 'hico-det', 
    'labels': [
        {'action_labels': 
            [{'human_part': 6, 'partstate': 0}, {'human_part': 9, 'partstate': 0}, {'human_part': 4, 'partstate': 0}, {'human_part': 0, 'partstate': 0}, {'human_part': 3, 'partstate': 0}], 'height': 480, 'hoi_id': 153, 'human_bbox': [208, 33, 427, 300], 'object_bbox': [59, 98, 572, 405], 'width': 640}, 
        {'action_labels': 
            [{'human_part': 4, 'partstate': 0}, {'human_part': 1, 'partstate': 6}, {'human_part': 2, 'partstate': 6}], 'height': 480, 'hoi_id': 156, 'human_bbox': [209, 26, 444, 317], 'object_bbox': [59, 99, 579, 395], 'width': 640}, 
        {'action_labels': 
            [{'human_part': 6, 'partstate': 0}, {'human_part': 9, 'partstate': 0}, {'human_part': 4, 'partstate': 0}, {'human_part': 0, 'partstate': 0}, {'human_part': 3, 'partstate': 0}], 'height': 480, 'hoi_id': 154, 'human_bbox': [213, 20, 438, 357], 'object_bbox': [77, 115, 583, 396], 'width': 640}, {'action_labels': [{'human_part': 4, 'partstate': 0}], 'height': 480, 'hoi_id': 155, 'human_bbox': [206, 33, 427, 306], 'object_bbox': [61, 100, 571, 401], 'width': 640}], 
        ...

    'path_prefix': 'hico_20160224_det/images/train2015'}


    """
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# FreiHAND""")
    return


@app.cell
def _():
    """
    training_K = json.load(open("./../_data/_usefull/FreiHAND/FreiHAND_pub_v2/training_K.json", "r"))
    training_mano = json.load(open("./../_data/_usefull/FreiHAND/FreiHAND_pub_v2/training_mano.json", "r"))
    training_scale = json.load(open("./../_data/_usefull/FreiHAND/FreiHAND_pub_v2/training_scale.json", "r"))
    training_verts = json.load(open("./../_data/_usefull/FreiHAND/FreiHAND_pub_v2/training_verts.json", "r"))
    training_xyz = json.load(open("./../_data/_usefull/FreiHAND/FreiHAND_pub_v2/training_xyz.json", "r"))

    # length is the same for all = 32560 = number of greenscreened images

    K = intrinsic camera matrix
    mano = mano annotations ?
    verts = 3d vertecies
    xyz = 3d shape
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Kinect""")
    return


@app.cell
def _():
    # info = json.load(open("./../_data/_usefull/KinectDatasets/data/training/info.json", "r"))
    # calib = json.load(open("./../_data/_usefull/KinectDatasets/data/training/calib.json", "r"))
    # anno = json.load(open("./../_data/_usefull/KinectDatasets/data/training/anno.json", "r"))
    # pred_sdk = json.load(open("./../_data/_usefull/KinectDatasets/data/training/pred_sdk.json", "r"))

    # captury_info = json.load(open("./../_data/_usefull/KinectDatasets/data/captury_train/info.json", "r"))
    # captury_calib = json.load(open("./../_data/_usefull/KinectDatasets/data/captury_train/calib.json", "r"))
    # captury_anno = json.load(open("./../_data/_usefull/KinectDatasets/data/captury_train/anno.json", "r"))
    # captury_pred_sdk = json.load(open("./../_data/_usefull/KinectDatasets/data/captury_train/pred_sdk.json", "r"))
    # captury_pred_sdk_cap = json.load(open("./../_data/_usefull/KinectDatasets/data/captury_train/pred_sdk_cap.json", "r"))

    """
    1920x1080

    anno[0][0] = 18 kpts + vis

    pred_sdk[0][i] = 25 kpts + vis (is a prediction)
    (i == 4 for all)


    captury_pred_sdk[0][i] = 25 kpts + vis
    (i == 1 for all)

    captury_pred_sdk_cap[0][i] = 25 kpts + vis
    (i == 2 for all)


    """
    None
    return


if __name__ == "__main__":
    app.run()
