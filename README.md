# Homemade Motion Capture

> The goal of this project is to provide a set of tools to setup, train and deploy a (mostly cheap) alternative to high end motion capturing intended for personal use.

## Feature Roadmap
### 0.1
- [ ] Person Detection
- [ ] Body Part Detection
### 0.2
- [ ] Human Mesh Recovery ([link](https://github.com/akanazawa/hmr))
- [ ] Human Instance segmentation (for preprocessing)
### 0.3
- [ ] Joint Projection/Extraction (double-quaternion, 6DoF)
- [ ] Unsupervised Human Motion Prior (MoCap data) for motion generation and pose/motion correction
### 0.4
- [ ] Create Constraint Rigidbody Humanoid Model
- [ ] RL Policy for Controlling Humanoid Rigid Body for "Dynamic Animation Retargeting"
### 0.5
- [ ] Face Detection
- [ ] 2D Face Keypoints
- [ ] Face Emotion Recognition
- [ ] Face Segmentation
- [ ] Eye Tracking ([link](https://github.com/JEOresearch/EyeTracker))
### 0.6
- [ ] Hand Detection
- [ ] 2D Hand Keypoints
- [ ] Hand Segmentation
### 0.7
- [ ] Create Constrained Rigidbody Face Model
- [ ] Create Constraint Rigidbody Hand Model

### Unclear
- [ ] Multi Camera Tracking
- [ ] Full Body Motion Classification
- [ ] Full Body Motion from Text
- [ ] Discrete hand pose estimation (if there even is a dataset for this)
- [ ] Multi cammera triangulation
- [ ] Object Detection
- [ ] Human object interactions

## Reading List
- [x] [Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector](https://docs.ultralytics.com/models/rtdetr/)
- [x] [Towards Accurate Faial Landmark Detection via Cascaded Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Accurate_Facial_Landmark_Detection_via_Cascaded_Transformers_CVPR_2022_paper.pdf)
- [x] [Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Xia_Sparse_Local_Patch_Transformer_for_Robust_Face_Alignment_and_Landmarks_CVPR_2022_paper.pdf)
- [x] [RePFormer: Refinement Pyramid Transformer for Robust Facial Landmark Detection](https://arxiv.org/pdf/2207.03917.pdf)
- [ ] [Revisiting Quantization Error in Face Alignment](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Lan_Revisting_Quantization_Error_in_Face_Alignment_ICCVW_2021_paper.pdf)
- [ ] [Learnable Triangulation of Human Pose](https://arxiv.org/pdf/1905.05754.pdf)
- [ ] [Shape Preserving Facial Landmarks with Graph Attention Networks](https://arxiv.org/pdf/2210.07233.pdf)
- [ ] [Vision Transformer with Deformable Attention](https://github.com/LeapLabTHU/DAT/tree/main)

## Acknowledgement
- A bunch of research projects all directely referenced by name with papers and links provided
- few code snippets used from [ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)
- ICP from [procrustes/ICP.py](https://github.com/bmershon/procrustes/blob/master/ICP.py)
- sam optimizer from [sam](https://github.com/davda54/sam/)
- WiseIoU from [WiseIoU](https://github.com/Instinct323/Wise-IoU)
- Facer from [FACER toolkit (FaRL models)](https://github.com/FacePerceiver/facer)
- Shape Preserving Facial Landmarks with Graph Attention Networks from [SPIGA](https://github.com/andresprados/spiga)
