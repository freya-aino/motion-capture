from copy import deepcopy
import warnings
from functools import reduce
import math
import gc
import os
import sys
import time
import traceback
import shutil
import glob
import cv2
import json
import csv
import numpy as np
import pandas as pd
import torch as T
from pathlib import Path

from ICP import stochasticICP_search


# -----------------------------------------------------------------------------------------------------------

def calculate_stereoscopic_coords(pl, pr):

    assert pl.shape[-1] >= 2 and pr.shape[-1] >= 2, "dim(-1) has to have at least 2 elements"

    orig_shape = pl.shape
    
    # print(T.prod(T.tensor(orig_shape[:-1])), orig_shape[-1])

    pl = pl.reshape(T.prod(T.tensor(orig_shape[:-1])).item(), orig_shape[-1])
    pr = pr.reshape(T.prod(T.tensor(orig_shape[:-1])).item(), orig_shape[-1])


    # X = INTER_CAM_DISTANCE * (pl[:, 0] - (CAMERA_OUT_WIDTH / 2)) / (pl[:, 0] - pr[:, 0])
    # Y = INTER_CAM_DISTANCE * FOCAL_LENGTH_W * (pl[:, 1] - (CAMERA_OUT_HEIGHT / 2)) / (FOCAL_LEGNTH_H * (pl[:, 0] - pr[:, 0]))
    X = pl[:, 0]
    Y = pl[:, 1]
    Z = INTER_CAM_DISTANCE * FOCAL_LENGTH_W / (pl[:, 0] - pr[:, 0])

    return T.stack([X, Y, Z], axis = -1).reshape(*[*orig_shape[:-1], 3])

    # cw = CAMERA_OUT_WIDTH * PIXEL_SIZE
    # focal_length = cw / 2 * (1 / (math.tan(FOV/2)))
    # orig_shape = pl.shape

    # xl = (pl[:, 0] - (cw / 2)) # / cw
    # xr = (pr[:, 0] - (cw / 2)) # / cw
    # disparity = (xl - xr)

    # z = T.abs(focal_length * INTER_CAM_DISTANCE / disparity)
    # x = pl[:, 0] * INTER_CAM_DISTANCE / disparity
    # y = pl[:, 1] * INTER_CAM_DISTANCE / disparity

    # return T.stack([x, y, z], axis = -1).reshape(*[*orig_shape[:-1], 3])


def undistort(img, crop = False):
    img_h, img_w = img.shape[:2]

    cam_mat = np.array([
        [FOCAL_LENGTH_W, 0.0,            (img_w) / 2], 
        [0.0,            FOCAL_LEGNTH_H, (img_h) / 2], 
        [0.0,            0.0,            1.0]
    ])

    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(cam_mat, DISTANCE_COEFFICIENTS, (img_w, img_h), 1, (img_w, img_h))
    dst = cv2.undistort(img, cam_mat, DISTANCE_COEFFICIENTS, None, new_cam_mat)

    if crop:
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (img_w, img_h), cv2.INTER_LINEAR)

    return dst


def load_all_frames(video_path, out_w = None, out_h = None):

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if out_w == None else out_w
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if out_h == None else out_h

    out = np.zeros((num_frames, h, w, 3), dtype = np.uint8)

    try:
        
        print("loading frames [", end = "")

        for ii in range(num_frames):

            print("|", end = "", flush = True)

            ret, frame = cap.read()
            assert ret, "VideoCapture is empty before end of sequence"

            if out_w == None or out_h == None:
                out[ii] = frame[:]
            else:
                out[ii] = cv2.resize(frame, (out_w, out_h), cv2.INTER_LINEAR)
                
                print(frame.shape)


            del frame
            gc.collect()
            time.sleep(0.001)

        out = np.float32(out)
        
        print("]")

    except:
        traceback.print_exc()
        print("terminating captures...")
        cap.release()
        print("]")
        exit(1)
        
    cap.release()
    return out



# OUTPUT MODEL -----------------------------------------------------------------------------------------------------------

# def moving_average_nan_interpolate(x, k, smooth_not_nan = True):

#     assert (k - 1) % 2 == 0, "bad kernel width"

#     sks_hlf = math.floor(k/2)

#     out = T.fill_(T.zeros_like(x), T.nan)

#     for i in range(x.shape[0]):
        
#         mi = max(i-sks_hlf, 0)
#         ma = min(i+sks_hlf+1, x.shape[0])
#         X = deepcopy(x[mi:ma])
#         # K = deepcopy(k[sks_hlf-min(sks_hlf, i):sks_hlf+1+min(sks_hlf, x.shape[0] - i - 1)])

#         # if X.isnan().any(-1).all(0).any()
        
        
#         X[(X == 0.).all(-1)] = T.nan

#         out[i] = T.nanmean(X, 0)

#         if not smooth_not_nan:
#             i_not_nan_mask = ~x[i].isnan().any(-1)
#             out[i][i_not_nan_mask] = x[i][i_not_nan_mask]
            
#     return out


# def interpolate_on_nan(kpts, smooth = False, not_nan_percentage_requirement = 0.5, smoothing_kernel_width = 3):

#     kpts = deepcopy(kpts)

#     assert (len(kpts[~kpts.isnan()]) / len(kpts)) > not_nan_percentage_requirement, "not nan keypoints have to be at least " + str(not_nan_percentage_requirement * 100) + " %"
#     assert smoothing_kernel_width >= 3, "smoothing kernel width has to be at least 3"

#     while kpts.isnan().any():
#         print("nan interpolation missing", kpts.isnan().any(-1).sum().item() / kpts.shape[1])
#         kpts = moving_average_nan_interpolate(kpts, smoothing_kernel_width, smooth_not_nan = False)
#     if smooth:
#         kpts = moving_average_nan_interpolate(kpts, smoothing_kernel_width, smooth_not_nan = True)

#     return kpts


# def interpolate_on_score(kpts, scores, cutof_score_quantile, smooth = False, not_nan_percentage_requirement = 0.5, smoothing_kernel_width = 3):

#     kpts = deepcopy(kpts)

#     assert (len(scores[~scores.isnan()]) / len(scores)) > not_nan_percentage_requirement, "non nan scores have to be at least " + str(not_nan_percentage_requirement * 100) + " %"
#     assert kpts.shape[0] == scores.shape[0], "kpts and scores have to have the same length"

#     score_nan_mask = scores.isnan()
#     score_mask = scores < scores[~scores.isnan()].quantile(cutof_score_quantile)
#     kpts[~score_nan_mask & score_mask] = T.nan

#     while kpts.isnan().any():
#         print("score interpolation missing", kpts.isnan().any(-1).sum().item() / kpts.shape[1])
#         kpts = moving_average_nan_interpolate(kpts, smoothing_kernel_width, smooth_not_nan = False)
#     if smooth:
#         kpts = moving_average_nan_interpolate(kpts, smoothing_kernel_width, smooth_not_nan = True)

#     return kpts


# def interpolate_on_spatial_variance(kpts_list, cutof_spatial_var_quantile, smooth = False, smoothing_kernel_width = 3):

#     kpts_list = deepcopy(kpts_list)

#     assert len(kpts_list) > 1, "at least two version of the same kpts have to be given"
#     kpts_len = [kpts.shape[0] for kpts in kpts_list]
#     assert all([kpts_len[0] == kpn for kpn in kpts_len]), "all inputs have to have the same length"
#     num_kpts = [kpts.shape[1] for kpts in kpts_list]
#     assert all([num_kpts[0] == kpn for kpn in num_kpts]), "all inputs have to have the same number of keypoints"
#     assert all([~(kpts.isnan().any()) for kpts in kpts_list]), "all keypoints have to be not nan"
#     assert smoothing_kernel_width >= 3, "smoothing kernel width has to be at least 3"
    
#     normed_kpts_list = [(kpts - kpts.mean(1).unsqueeze(1)) / kpts.std(1).unsqueeze(1) for kpts in kpts_list]
#     total_var = T.stack(normed_kpts_list).var(0).sum(-1)
#     total_mask = (total_var >= total_var.quantile(cutof_spatial_var_quantile))

#     for i in range(len(kpts_list)):
#         kpts_list[i][total_mask] = T.nan

#         while kpts_list[i].isnan().any():
#             print("spatial variance interpolation", i, "missing", kpts_list[i].isnan().any(-1).sum().item() / kpts_list[i].shape[1])
#             kpts_list[i] = moving_average_nan_interpolate(kpts_list[i], smoothing_kernel_width, smooth_not_nan = False)
#         if smooth:
#             kpts_list[i] = moving_average_nan_interpolate(kpts_list[i], smoothing_kernel_width, smooth_not_nan = True)

#     return kpts_list


# def interpolate_on_temporal_variance(kpts, temporal_variance_kernel_width, cutof_temporal_var_quantile, smooth = False, smoothing_kernel_width = 3):

#     kpts = deepcopy(kpts)

#     assert not kpts.isnan().any(), "all keypoints have to be not nan"
#     assert smoothing_kernel_width >= 3, "smoothing kernel width has to be at least 3"
#     assert (temporal_variance_kernel_width - 1) % 2 == 0, "variance kernel should be W - 1 % == 0"

#     vkw_hlf = (temporal_variance_kernel_width-1) // 2

#     total_var = T.zeros(kpts.shape[0], kpts.shape[1])
#     for i in range(kpts.shape[0]):
#         mi = max(i-vkw_hlf, 0)
#         ma = min(i+vkw_hlf+1, kpts.shape[0])
#         total_var[i] = kpts[mi:ma].var(0).sum(-1)

#     kpts[(total_var >= total_var.quantile(cutof_temporal_var_quantile))] = T.nan

#     while kpts.isnan().any():
#         print("temporal variance interpolation missing", kpts.isnan().any(-1).sum().item() / kpts.shape[1])
#         kpts = moving_average_nan_interpolate(kpts, smoothing_kernel_width, smooth_not_nan = False)
#     if smooth:
#         kpts = moving_average_nan_interpolate(kpts, smoothing_kernel_width, smooth_not_nan = True)

#     return kpts



def prepare_osf(x, face_anchor_offset = 0.6):

    x.loc[x["Success"].isna(), "Success"] = 0.
    x["Success"] = x["Success"].astype(bool)
    
    availability = T.tensor(reduce(lambda a, b: a & b, [x["Success"], *[~x[k].isna() for k in x.keys()]]), dtype = T.bool)

    kpts = T.stack([T.tensor(list(x["Keypoints"].iloc[i])).reshape(68, 3) if availability[i] else T.nan * T.ones(68, 3) for i in range(len(x))]).to(dtype = T.float32)
    kpts_2d = kpts[..., :2] # .flip(-1)
    kpts_confidences = kpts[..., 2]


    kpts_3d_image_space = T.stack([T.tensor(list(x["Keypoints_3D"].iloc[i])).reshape(70, 3) if availability[i] else T.nan * T.ones(70, 3) for i in range(len(x))]).to(dtype = T.float32)


    static_face_ancor = kpts_3d_image_space[:, 27:36, :].nanmean(1, keepdim = True) - T.tensor([0, 0, face_anchor_offset])
    kpts_3d_image_space = T.cat([kpts_3d_image_space, static_face_ancor], dim = 1)
    kpts_3d_local_space = deepcopy(kpts_3d_image_space)

    # T.tensor(info[x[c]["Frame"] == jj].iloc[0])) if availability[jj] else T.nan * T.ones(27) for jj in range(num_frames)]).to(dtype = T.float32)
    # bbox = T.stack([T.tensor(list(x["Bbox"].iloc[i])) if availability[i] else T.nan * T.ones(4) for i in range(len(x))]).to(dtype = T.float32)
    translation = T.stack([T.tensor(list(x["Translation"].iloc[i])) if availability[i] else T.nan * T.ones(3) for i in range(len(x))]).to(dtype = T.float32)
    # euler = T.stack([T.tensor(list(x["Euler"].iloc[i])) if availability[i] else T.nan * T.ones(3) for i in range(len(x))]).to(dtype = T.float32)
    quaternion = T.stack([T.tensor(list(x["Quaternion"].iloc[i])) if availability[i] else T.nan * T.ones(4) for i in range(len(x))]).to(dtype = T.float32)


    from scipy.spatial.transform import Rotation as sci_rot

    camera_rot = np.array([[CAMERA_OUT_WIDTH, 0, CAMERA_OUT_WIDTH/2], [0, CAMERA_OUT_WIDTH, CAMERA_OUT_HEIGHT/2], [0, 0, 1]], dtype = np.float32)

    for i in range(kpts_3d_image_space.shape[0]):
        if availability[i]:
            pt_3d = kpts_3d_image_space[i] # , [*range(0, 66), 68, 69]]
            
            rot_mat_q = sci_rot.from_quat(quaternion[i].numpy()).as_matrix()
            pt_3d[:, 0] = -pt_3d[:, 0]
            pt_3d = pt_3d.numpy().dot(rot_mat_q)
            pt_3d = pt_3d + translation[i].numpy()
            pt_3d = pt_3d.dot(camera_rot.transpose())
            pt_3d = T.tensor(pt_3d, dtype = T.float32)

            depth = pt_3d[:, 2].unsqueeze(-1)
            depth[depth == 0] = 0.0001
            pt_3d[:, :2] = pt_3d[:, :2] / depth
            # pt_3d[:, :2] = kpts_2d[i]

            kpts_3d_image_space[i, :] = pt_3d[:]


    kpts_3d_image_space[..., :2] = kpts_3d_image_space[..., :2].flip(-1)
    kpts_2d[..., :2] = kpts_2d[..., :2].flip(-1)

    return {
        "availability": availability,
        "confidences": T.tensor([x["Conf"].iloc[i] if availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32),
        "keypoints_2d": kpts_2d,
        "keypoints_confidences": kpts_confidences,
        "keypoints_3d_image_space": kpts_3d_image_space,
        "keypoints_3d_local_space": kpts_3d_local_space,
        # "info": T.cat([bbox, translation, euler], dim = -1),
    }


def prepare_mediapipe(x):

    right_mask_list = [~x["Keypoints_right_hand"].isna(), ~x["Score_right_hand"].isna()]
    right_availability = T.tensor(reduce(lambda a, b: a & b, right_mask_list), dtype = T.bool)
    
    left_mask_list = [~x["Keypoints_left_hand"].isna(), ~x["Score_left_hand"].isna()]
    left_availability = T.tensor(reduce(lambda a, b: a & b, left_mask_list), dtype = T.bool)

    
    scores_right_hand = T.tensor([x["Score_right_hand"].iloc[i] if right_availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32)
    scores_left_hand = T.tensor([x["Score_left_hand"].iloc[i] if left_availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32)

    keypoints_right_hand = T.stack([T.tensor(list(x["Keypoints_right_hand"].iloc[i])).reshape(21, 3) if right_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_left_hand = T.stack([T.tensor(list(x["Keypoints_left_hand"].iloc[i])).reshape(21, 3) if left_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)

    keypoints_right_hand_local = T.stack([T.tensor(list(x["Keypoints_right_hand_local"].iloc[i])).reshape(21, 3) if right_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_left_hand_local = T.stack([T.tensor(list(x["Keypoints_left_hand_local"].iloc[i])).reshape(21, 3) if left_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)

    # keypoints_right_hand[:, 0] = keypoints_right_hand[:, 0] * 1920
    # keypoints_right_hand[:, 1] = keypoints_right_hand[:, 1] * 1080

    return {
        "availability_right_hand": right_availability,
        "availability_left_hand": left_availability,
        "scores_right_hand": scores_right_hand,
        "scores_left_hand": scores_left_hand,
        "keypoints_right_hand": keypoints_right_hand,
        "keypoints_left_hand": keypoints_left_hand,
        "keypoints_right_hand_local_space": keypoints_right_hand_local,
        "keypoints_left_hand_local_space": keypoints_left_hand_local,
    }




# def prepare_vtp3d(x):
#     availability = T.tensor((~x["Score"].isna()) & (~x["Keypoints"].isna()), dtype = T.bool)
#     return {
#         "availability": availability,
#         "keypoints": T.stack([T.tensor(list(x["Keypoints"].iloc[i])).reshape(17, 3) if availability[i] else T.nan * T.ones(17, 3) for i in range(len(x))]).to(dtype = T.float32),
#         "scores": T.tensor([x["Score"].iloc[i] if availability[i] else T.nan for i in range(len(x))], dtype = T.float32)
#     }

# def prepare_op(x):

#     availability = T.tensor(reduce(lambda a, b: a & b, [~x[k].isna() for k in x.keys()]), dtype = T.bool)
    
#     keypoints_pose = T.stack([T.tensor(list(x["Keypoints_pose"].iloc[i])).reshape(25, 2) if availability[i] else T.nan * T.ones(25, 2) for i in range(len(x))]).to(dtype = T.float32)
#     keypoints_face = T.stack([T.tensor(list(x["Keypoints_face"].iloc[i])).reshape(70, 2) if availability[i] else T.nan * T.ones(70, 2) for i in range(len(x))]).to(dtype = T.float32)
#     keypoints_left_hand = T.stack([T.tensor(list(x["Keypoints_left_hand"].iloc[i])).reshape(21, 2) if availability[i] else T.nan * T.ones(21, 2) for i in range(len(x))]).to(dtype = T.float32)
#     keypoints_right_hand = T.stack([T.tensor(list(x["Keypoints_right_hand"].iloc[i])).reshape(21, 2) if availability[i] else T.nan * T.ones(21, 2) for i in range(len(x))]).to(dtype = T.float32)

#     scores_pose = T.stack([T.tensor(list(x["Scores_pose"].iloc[i])) if availability[i] else T.nan * T.ones(25) for i in range(len(x))]).to(dtype = T.float32)
#     scores_face = T.stack([T.tensor(list(x["Scores_face"].iloc[i])) if availability[i] else T.nan * T.ones(70) for i in range(len(x))]).to(dtype = T.float32)
#     scores_left_hand = T.stack([T.tensor(list(x["Scores_left_hand"].iloc[i])) if availability[i] else T.nan * T.ones(21) for i in range(len(x))]).to(dtype = T.float32)
#     scores_right_hand = T.stack([T.tensor(list(x["Scores_right_hand"].iloc[i])) if availability[i] else T.nan * T.ones(21) for i in range(len(x))]).to(dtype = T.float32)

#     visibilities_pose = ~(keypoints_pose == 0.).all(-1)
#     visibilities_face = ~(keypoints_face == 0.).all(-1)
#     visibilities_left_hand = ~(keypoints_left_hand == 0.).all(-1)
#     visibilities_right_hand = ~(keypoints_right_hand == 0.).all(-1)

#     keypoints_pose[~visibilities_pose] = T.nan
#     keypoints_face[~visibilities_face] = T.nan
#     keypoints_left_hand[~visibilities_left_hand] = T.nan
#     keypoints_right_hand[~visibilities_right_hand] = T.nan

#     print("TODO: check if face keypoints lsat 5 elements are usefull")

#     return {
#         "availability": availability,
#         "keypoints_pose": keypoints_pose,
#         "keypoints_face": keypoints_face,
#         "keypoints_left_hand": keypoints_left_hand,
#         "keypoints_right_hand": keypoints_right_hand,
#         "scores_pose": scores_pose,
#         "scores_face": scores_face,
#         "scores_left_hand": scores_left_hand,
#         "scores_right_hand": scores_right_hand,
#         "visibilities_pose": visibilities_pose,
#         "visibilities_face": visibilities_face,
#         "visibilities_left_hand": visibilities_left_hand,
#         "visibilities_right_hand": visibilities_right_hand,
#     }

# def prepare_det2_vp3d(x):
    
#     availability = T.tensor(reduce(lambda a, b: a & b, [~x[k].isna() for k in x.keys()]), dtype = T.bool)
    
#     bboxes = T.stack([T.tensor(list(x["Bbox"].iloc[i])) if availability[i] else T.nan * T.ones(4) for i in range(len(x))]).to(dtype = T.float32)
#     keypoints = T.stack([T.tensor(list(x["Keypoints"].iloc[i])).reshape(17, 3) if availability[i] else T.nan * T.ones(17, 3) for i in range(len(x))]).to(dtype = T.float32)
#     scores = T.tensor([x["Score"].iloc[i] if availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32)

#     return {
#         "availability": availability,
#         "bboxes": bboxes,
#         "keypoints": keypoints,
#         "scores": scores
#     }




def load_3d_keypoints(vid_name): # , det2_vp3d = False):
    

    osf_cam_0_y = prepare_osf(pd.read_json(os.path.join(ANNOTATION_PATH, "OPEN_SEE_FACE", vid_name, "0.json")))
    osf_cam_1_y = prepare_osf(pd.read_json(os.path.join(ANNOTATION_PATH, "OPEN_SEE_FACE", vid_name, "1.json")))
    
    mp_cam_0_y = prepare_mediapipe(pd.read_json(os.path.join(ANNOTATION_PATH, "MEDIAPIPE", vid_name, "0.json")))
    mp_cam_1_y = prepare_mediapipe(pd.read_json(os.path.join(ANNOTATION_PATH, "MEDIAPIPE", vid_name, "1.json")))


    # restrict length
    min_length = min([
        *[osf_cam_0_y[k].shape[0] for k in osf_cam_0_y], *[osf_cam_1_y[k].shape[0] for k in osf_cam_1_y],
        *[mp_cam_0_y[k].shape[0] for k in mp_cam_0_y], *[mp_cam_1_y[k].shape[0] for k in mp_cam_1_y]])
    for k in osf_cam_0_y.keys():
        osf_cam_0_y[k] = osf_cam_0_y[k][:min_length]
    for k in osf_cam_1_y.keys():
        osf_cam_1_y[k] = osf_cam_1_y[k][:min_length]
    for k in mp_cam_0_y.keys():
        mp_cam_0_y[k] = mp_cam_0_y[k][:min_length]
    for k in mp_cam_1_y.keys():
        mp_cam_1_y[k] = mp_cam_1_y[k][:min_length]

    
    # get availability masks (not none)
    cam_0_filter_list = {
        "osf": osf_cam_0_y["availability"],
        "mpr": mp_cam_0_y["availability_right_hand"],
        "mpl": mp_cam_0_y["availability_left_hand"],
    }
    cam_1_filter_list = {
        "osf": osf_cam_1_y["availability"],
        "mpr": mp_cam_1_y["availability_right_hand"],
        "mpl": mp_cam_1_y["availability_left_hand"],
    }


    return {
        "cam_0": {
            "mp": mp_cam_0_y,
            "osf": osf_cam_0_y,
            "availability_masks": cam_0_filter_list,
        },
        "cam_1": {
            "mp": mp_cam_1_y,
            "osf": osf_cam_1_y,
            "availability_masks": cam_1_filter_list,
        }
    }

def osf_kpts_prep(
    osf_image_space,
    osf_local_space,
    icp_max_iters = 10,
    icp_samples = 30):

    osf_gn = ms_norm(osf_image_space, dim = 0)
    osf_ln = ms_norm(osf_local_space, dim = 0) 
    osf_valid_mask = ~osf_gn.isnan().any(-1) & ~osf_ln.isnan().any(-1)

    if osf_valid_mask.any():
        osf_ln[osf_valid_mask] = T.tensor(stochasticICP_search(osf_ln[osf_valid_mask].numpy().T, osf_gn[osf_valid_mask].numpy().T, icp_max_iters, icp_samples), dtype=T.float32)
    # osf_ln = T.tensor(stochasticICP_search(osf_ln.numpy().T, osf_gn.numpy().T, icp_max_iters, icp_samples), dtype=T.float32)

    osf_ln = ms_norm(osf_ln, dim = 0)
    osf_ln = osf_ln - osf_ln[78] # 79 = face anchor

    return osf_ln

def mp_kpts_prep(
    mp_image_space,
    mp_local_space,
    icp_max_iters = 10,
    icp_samples = 30):

    mp_gn = ms_norm(mp_image_space, dim = 0)
    mp_ln = ms_norm(mp_local_space, dim = 0) 
    mp_valid_mask = ~mp_gn.isnan().any(-1) & ~mp_ln.isnan().any(-1)

    if mp_valid_mask.any():
        mp_ln[mp_valid_mask] = T.tensor(stochasticICP_search(mp_ln[mp_valid_mask].numpy().T, mp_gn[mp_valid_mask].numpy().T, icp_max_iters, icp_samples), dtype=T.float32)
    # mp_ln = T.tensor(stochasticICP_search(mp_ln.numpy().T, mp_gn.numpy().T, icp_max_iters, icp_samples), dtype=T.float32)

    mp_ln = ms_norm(mp_ln, dim = 0)
    mp_ln = mp_ln - mp_ln[0]  # 0 = wrist

    return mp_ln


def get_bbox_from_kpts(kpts, baseline_distance = 0.1, normalize = True):

    if kpts.isnan().any():
        return T.ones(4) * T.nan

    kpts = kpts[..., :2]
    kpts = kpts[~(kpts.isnan().any(-1))]

    min_ = kpts.min(0).values
    max_ = kpts.max(0).values
    
    x, y = min_
    h, w = max_ - min_

    inverse_factor_w = (CAMERA_OUT_WIDTH - w) * baseline_distance
    inverse_factor_h = 1 - (h / CAMERA_OUT_HEIGHT)

    x = x - inverse_factor_w
    y = y - inverse_factor_h
    h = h + inverse_factor_h
    w = w + inverse_factor_w

    if normalize:
        x = x / CAMERA_OUT_HEIGHT
        y = y / CAMERA_OUT_WIDTH
        h = h / CAMERA_OUT_HEIGHT
        w = w / CAMERA_OUT_WIDTH

    return T.stack([y, x, h, w])



# ANNOTATIONS AUTOMATIONS -----------------------------------------------------------------------------------------------------------

def remove_file(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def empty_directory(path):
    for i in glob.glob(os.path.join(path, "*")):
        remove_file(i)


def get_video_annotation_status(video_name):
    assert video_name in os.listdir(VIDEO_PATH), "video " + video_name + " has to be in " + str(VIDEO_PATH)

    annotation_names = os.listdir(ANNOTATION_PATH)
    anno_info = dict([(ann_name, video_name in os.listdir(os.path.join(ANNOTATION_PATH, ann_name))) for ann_name in annotation_names if os.path.isdir(os.path.join(ANNOTATION_PATH, ann_name))])
    return anno_info


def anno_wrapper(func):
    def wrap(*args, **kwargs):

        assert "service_name" in kwargs, "service_name has to be given as kwarg"
        assert "video_name" in kwargs, "video_name has to be given as kwarg"

        local_anno_path = os.path.join(ANNOTATION_PATH, kwargs["service_name"], kwargs["video_name"])
        if not os.path.exists(local_anno_path):
            print("creating", local_anno_path)
            os.mkdir(local_anno_path)

        try: 
            print("----------------  ANNOTATE", kwargs["video_name"], "WITH", kwargs["service_name"], " ----------------")
            func(*args, **kwargs)
            print("---------------------------------  DONE ---------------------------------")
        except:
            traceback.print_exc()
            time.sleep(1)
            print("deleting: ", local_anno_path)
            if os.path.exists(local_anno_path):
                shutil.rmtree(local_anno_path)
            exit()

    return wrap


@anno_wrapper
def detectron2_video_pose_3d_annotate(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    warnings.filterwarnings("ignore")
    
    from detectron2.engine import DefaultPredictor
    from detectron2.config.config import get_cfg

    config_file_path = str(Path(__file__).absolute().parent.joinpath("external_repo_cache", "VideoPose3DwithDetectron2", "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    model_file_path = str(Path(__file__).absolute().parent.joinpath("external_repo_cache", "VideoPose3DwithDetectron2", "model_final_5ad38f.pkl"))

    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)


    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):
        cam_id = cam_name.split(".")[0]

        video_frames = load_all_frames(os.path.join(VIDEO_PATH, video_name, cam_name))

        out = []

        print("starting inference on:", os.path.join(VIDEO_PATH, video_name, cam_name), "...")
        print("annotating     [", end = "")
        for frame in video_frames:

            pose_output = predictor(frame)

            print("|", end = "", flush = True)


            if len(pose_output["instances"].pred_boxes) > 0:
                out.append({
                    "Bbox": list(pose_output["instances"].pred_boxes[0].tensor.flatten().cpu().numpy()),
                    "Score": pose_output["instances"].scores[0].cpu().item(),
                    "Keypoints": list(pose_output["instances"].pred_keypoints[0].flatten().cpu().numpy())
                })
            else:
                out.append({"Bbox": None, "Score": None, "Keypoints": None})

        print("\nsaving outputs to:", os.path.join(local_anno_path, cam_id + ".json"), "...")
        out = pd.DataFrame.from_records(out)
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))


    warnings.filterwarnings("default")



@anno_wrapper
def mediapipe_annotation(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    from mediapipe.python.solutions.hands import Hands

    hand_tracker = Hands()

    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):
        cam_id = cam_name.split(".")[0]

        video_frames = np.uint8(load_all_frames(os.path.join(VIDEO_PATH, video_name, cam_name)))

        out = []

        print("annotating     [", end = "")
        for frame in video_frames:

            print("|", end = "", flush = True)

            # frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hand_tracker.process(frame)

            oo = {
                "Score_right_hand": None,
                "Score_left_hand": None,
                "Keypoints_right_hand": None,
                "Keypoints_left_hand": None,
                "Keypoints_left_hand_local": None,
                "Keypoints_right_hand_local": None,
            }

            if result.multi_handedness != None:
                for i in range(len(result.multi_handedness)):
                    if result.multi_handedness[i].classification[0].label == "Right":
                        oo["Score_left_hand"] = result.multi_handedness[i].classification[0].score
                        oo["Keypoints_left_hand"] = list(np.array([[l.x * CAMERA_OUT_WIDTH, l.y * CAMERA_OUT_HEIGHT, l.z] for l in result.multi_hand_landmarks[i].landmark]).flatten())
                        oo["Keypoints_left_hand_local"] = list(np.array([[l.x, l.y, l.z] for l in result.multi_hand_world_landmarks[i].landmark]).flatten())
                    elif result.multi_handedness[i].classification[0].label == "Left":
                        oo["Score_right_hand"] = result.multi_handedness[i].classification[0].score
                        oo["Keypoints_right_hand"] = list(np.array([[l.x * CAMERA_OUT_WIDTH, l.y * CAMERA_OUT_HEIGHT, l.z] for l in result.multi_hand_landmarks[i].landmark]).flatten())
                        oo["Keypoints_right_hand_local"] = list(np.array([[l.x, l.y, l.z] for l in result.multi_hand_world_landmarks[i].landmark]).flatten())
            
            out.append(oo)

        print("\nsaving outputs to:", os.path.join(local_anno_path, cam_id + ".json"), "...")
        out = pd.DataFrame.from_records(out)
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))




@anno_wrapper
def open_see_face_annotate(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    proj_path = Path(__file__).absolute().parent.joinpath("external_repo_cache", "OpenSeeFace_altered")
    sys.path.append(str(proj_path))

    from external_repo_cache.OpenSeeFace_altered.tracker import Tracker as open_see_face_tracker

    model_dir = os.path.join(proj_path, "models")

    face_tracker = open_see_face_tracker(
        CAMERA_OUT_WIDTH, 
        CAMERA_OUT_HEIGHT, 
        model_type = 4,
        model_dir = model_dir, 
        silent = True)

    
    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):

        print("annotating video", os.path.join(VIDEO_PATH, video_name, cam_name))
        
        cam_id = cam_name.split(".")[0]
        v_path = os.path.join(VIDEO_PATH, video_name, cam_name)
        frames = load_all_frames(v_path)
        out = []
        

        for i in range(frames.shape[0]):

            oo = {
                "Frame": i,
                "Success": None,
                "Conf": None,
                "Keypoints": None,
                "Quaternion": None,
                "Euler": None,
                "Translation": None,
                "Bbox": None,
                "Eye_r": None,
                "Eye_r_conf": None,
                "Eye_l": None,
                "Eye_l_conf": None
            }

            face_result = face_tracker.predict(frames[i])

            if len(face_result) > 0:

                face_result = face_result[0]
                if face_result != None:

                    success = face_result.success
                    lms = face_result.lms.flatten()
                    lms_3d = face_result.pts_3d.flatten()
                    quaternion = face_result.quaternion
                    euler = face_result.euler
                    translation = face_result.translation
                    bbox = face_result.bbox
                    conf = face_result.conf

                    _, eye_r_y, eye_r_x, eye_r_conf = face_result.eye_state[0]
                    _, eye_l_y, eye_l_x, eye_l_conf = face_result.eye_state[1]

                    out.append({
                        "Frame": i,
                        "Success": success,
                        "Conf": conf,
                        "Keypoints": lms,
                        "Keypoints_3D": lms_3d,
                        "Quaternion": quaternion,
                        "Euler": euler,
                        "Translation": translation,
                        "Bbox": bbox,
                        "Eye_r": [eye_r_x, eye_r_y],
                        "Eye_r_conf": eye_r_conf,
                        "Eye_l": [eye_l_x, eye_l_y],
                        "Eye_l_conf": eye_l_conf
                    })
                else:
                    out.append(oo)

        out = pd.DataFrame.from_records(out)

        print("saving to", os.path.join(local_anno_path, cam_id + ".json"))
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))

    
@anno_wrapper
def openpose_annotate(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    proj_path = Path(__file__).absolute().parent.joinpath("external_repo_cache", "openpose")


    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):

        cam_id = cam_name.split(".")[0]

        print("annotating video and saving results...")

        cmd = [
            "cd " + str(proj_path) + " &",
            ".\\build\\x64\\Release\\OpenPoseDemo.exe",
            "--video " + os.path.join(VIDEO_PATH, video_name, cam_name),
            "--write_json " + os.path.join(local_anno_path, cam_id),
            "--face",
            "--hand",
            "--display 0",
            "--render_pose 0",
            "--number_people_max 1"
        ]
        os.system(" ".join(cmd))

        print("format json ...")
        out = []
        for ji in os.listdir(os.path.join(local_anno_path, cam_id)):
            
            frame = int(ji.split("_")[1])

            # try:
            with open(os.path.join(local_anno_path, cam_id, ji), "r+") as ff:
                J = json.load(ff)
            # except Exception as e:
            #     if e.__class__ != json.decoder.JSONDecodeError:
            #         traceback.print_exc()
            #         exit()
            #     else:
            #         print("frame", frame, "data unavailable")
            #         out.append({
            #             "Frame": frame,
            #             "Keypoints_pose": None,
            #             "Keypoints_face": None,
            #             "Keypoints_left_hand": None,
            #             "Keypoints_hand_right": None
            #         })
            #         continue
            

            if "people" in J and len(J["people"]) > 0:
                pose_keypoints_2d = np.array(J["people"][0]["pose_keypoints_2d"]).reshape(25, 3) if "pose_keypoints_2d" in J["people"][0] else None
                face_keypoints_2d = np.array(J["people"][0]["face_keypoints_2d"]).reshape(70, 3) if "face_keypoints_2d" in J["people"][0] else None
                hand_left_keypoints_2d = np.array(J["people"][0]["hand_left_keypoints_2d"]).reshape(21, 3) if "hand_left_keypoints_2d" in J["people"][0] else None
                hand_right_keypoints_2d = np.array(J["people"][0]["hand_right_keypoints_2d"]).reshape(21, 3) if "hand_right_keypoints_2d" in J["people"][0] else None
                
                pose_keypoints_2d, pose_scores_2d = (pose_keypoints_2d[:, :2].flatten(), pose_keypoints_2d[:, 2].flatten()) if type(pose_keypoints_2d) == np.ndarray else (None, None)
                face_keypoints_2d, face_scores_2d = (face_keypoints_2d[:, :2].flatten(), face_keypoints_2d[:, 2].flatten()) if type(face_keypoints_2d) == np.ndarray else (None, None)
                hand_left_keypoints_2d, hand_left_scores_2d = (hand_left_keypoints_2d[:, :2].flatten(), hand_left_keypoints_2d[:, 2].flatten()) if type(hand_left_keypoints_2d) == np.ndarray else (None, None)
                hand_right_keypoints_2d, hand_right_scores_2d = (hand_right_keypoints_2d[:, :2].flatten(), hand_right_keypoints_2d[:, 2].flatten()) if type(hand_right_keypoints_2d) == np.ndarray else (None, None)

            else:
                pose_keypoints_2d, pose_scores_2d = None, None
                face_keypoints_2d, face_scores_2d = None, None
                hand_left_keypoints_2d, hand_left_scores_2d = None, None
                hand_right_keypoints_2d, hand_right_scores_2d = None, None

            out.append({
                "Frame": frame,
                "Keypoints_pose": pose_keypoints_2d,
                "Scores_pose": pose_scores_2d,
                "Keypoints_face": face_keypoints_2d,
                "Scores_face": face_scores_2d,
                "Keypoints_left_hand": hand_left_keypoints_2d,
                "Scores_left_hand": hand_left_scores_2d,
                "Keypoints_right_hand": hand_right_keypoints_2d,
                "Scores_right_hand": hand_right_scores_2d
            })
        out = pd.DataFrame.from_records(out)
        fit_table = pd.DataFrame.from_records([{"Frame": i} for i in range(max(out["Frame"]))])
        out = pd.merge(fit_table, out, how = "left", on = "Frame")

        print("saving as ", os.path.join(local_anno_path, cam_id + ".json"), "...")
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))

        print("cleanup ...")
        shutil.rmtree(os.path.join(local_anno_path, cam_id))

        print("successfully annotated:", video_name + "\\" + cam_name)


@anno_wrapper
def video_to_pose_3d_annotate(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    proj_outputs_path = Path(__file__).absolute().parent.joinpath("external_repo_cache", "video_to_pose3D", "outputs")
    sys.path.append(str(proj_outputs_path.parent))

    from external_repo_cache.video_to_pose3D.videopose import inference_video


    print("emptying outputs:", os.path.join(proj_outputs_path), "...")
    empty_directory(os.path.join(proj_outputs_path))

    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):

        cam_id = cam_name.split(".")[0]


        print("copy video localy ...")
        shutil.copy2(os.path.join(VIDEO_PATH, video_name, cam_name), os.path.join(proj_outputs_path, cam_name))

        print("infere video ...")
        inference_video(os.path.join(proj_outputs_path, cam_name), "alpha_pose", proj_outputs_path)

        print("format json ...")
        out_json = os.path.join(proj_outputs_path, "alpha_pose_" + cam_id, "alphapose-results.json")
        out = pd.DataFrame.from_records(pd.read_json(out_json))
        out["image_id"] = out["image_id"].apply(lambda r: int(r.split(".")[0]))
        duplicate_kpts = [v for (v, c) in out["image_id"].value_counts().items() if c > 1]
        mask_list = [out["image_id"] != dub_kpt for dub_kpt in duplicate_kpts]
        out = out[reduce(lambda a, b: a & b, mask_list)] if len(mask_list) > 0 else out
        fit_table = pd.DataFrame.from_records([{"image_id": i} for i in range(max(out["image_id"]))])
        out = pd.merge(fit_table, out, how = "left", on = "image_id")
        out = out.rename(columns = {
            "image_id": "Frame", 
            "keypoints": "Keypoints", 
            "score": "Score"
        })
        out = out.drop(columns=["category_id"])

        print("save result ...")
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))

        print("cleanup ...")
        empty_directory(proj_outputs_path)

        print("check cleanup ...")
        assert len(os.listdir(proj_outputs_path)) == 0, ".\\outputs should be empty at this point but has: " + str(os.listdir(proj_outputs_path))
        
        print("successfully annotated:", video_name + "\\" + cam_name)




