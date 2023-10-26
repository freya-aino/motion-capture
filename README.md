# Homemade Motion Capture

> The goal of this project is to provide a set of tools to setup, train and deploy a (mostly cheap) alternative to high end motion capturing intended for personal use.

> ! The repository contains some legacy code, mainly in the core folder

## Versions & Roadmap
### 0.1
- [x] Basic Pipeline setup
  - [x] Test FastAPI
  - [x] Test Redis
  - [x] Test ZMQ
- [ ] Data
  - [x] Prepare dataloader for WIDER Face
  - [ ] Prepare dataloader for {todo: another face detection dataset}
  - [ ] Prepare dataloader for {todo: one image understanding dataset}
- [ ] Camera Calibration (only intrinsic calibration required)
- [ ] Reading
  - [ ] Read [Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector](https://docs.ultralytics.com/models/rtdetr/)
  - [ ] Read [Towards Accurate Faial Landmark Detection via Cascaded Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Accurate_Facial_Landmark_Detection_via_Cascaded_Transformers_CVPR_2022_paper.pdf)
  - [ ] Read [Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Xia_Sparse_Local_Patch_Transformer_for_Robust_Face_Alignment_and_Landmarks_CVPR_2022_paper.pdf)
  - [ ] Read [RePFormer: Refinement Pyramid Transformer for Robust Facial Landmark Detection](https://arxiv.org/pdf/2207.03917.pdf)
  - [ ] Read [Revisiting Quantization Error in Face Alignment](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Lan_Revisting_Quantization_Error_in_Face_Alignment_ICCVW_2021_paper.pdf)
  - [ ] Read [Shape Preserving Facial Landmarks with Graph Attention Networks](https://arxiv.org/pdf/2210.07233.pdf)
- [ ] Detached Inference from 3. party (deployment)
  - [ ] Implement detached inference with [FACER toolkit (FaRL models)](https://github.com/FacePerceiver/facer)
  - [ ] Implement detached inference with [SPIGA](https://github.com/andresprados/spiga) (Shape Preserving Facial Landmarks with Graph Attention Networks)
- [ ] Face Tracking Module
  - [x] Construct a basic Vision attention based module with alterations (Backbone, Head)
  - [ ] Write the training experiment for pretraining the Backbone
  - [ ] Write the training experiment for training the neck/head on face recognition
  - [ ] Train and confirm, different sizes and maybe adjust the model structure
### 0.2
- [ ] Face Keypoint Data
  - [x] Prepare dataloader for WFLW
  - [ ] Prepare dataloader for COFW
  - [ ] Prepare dataloader for FDDB
  - [ ] Annotate self recorded videos with help of the teacher networks
  - [ ] Format the face keypoint data relative to the face recognition bounding boxes (+ some padding)
- [ ] Face Keypoint Estimator
  - [ ] Write the training experiment for training a neck/head on keypoint estimation
  - [ ] Answer the question: in what way is my own recorded dataset semantically constraint that finetuning on it improves results, and to what extend ?
### 0.3
- [ ] Hand tracking
### 0.5
- [ ] Facial emotion recognition
- [ ] Discrete hand pose estimation (if there even is a dataset for this)
### 0.8
- [ ] human instance segmentation
- [ ] Face segmentation
### 1.0
- [ ] Human body keypoint estimation
- [ ] Multi cammera triangulation
### 1.5
- [ ] Human body part segmentation
### 2.0
- [ ] Object detection & 6DOF tracking
- [ ] Human object interactions


## Acknowledgement
- A bunch of research projects all directely referenced by name with papers and links provided
- few code snippets used from [ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)
- ICP from [procrustes/ICP.py](https://github.com/bmershon/procrustes/blob/master/ICP.py)