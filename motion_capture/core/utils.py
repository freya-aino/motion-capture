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
import torch as T

from pathlib import Path
from logging import getLogger
from ICP import stochasticICP_search

# -----------------------------------------------------------------------------------------------------------

logger = getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------

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



