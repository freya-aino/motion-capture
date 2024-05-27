import gc
import sys
import time
import traceback
import cv2
import json
import multiprocessing as mp
import torch as T
import numpy as np
from torch.utils import data
import pickle

from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from quaternions import Quaternion

# -----------------------------------------------------------------

'''
Structure & Primitives

would be extracted into respective models for
1) datasets
2) skeletons / rigs
3) export

'''

@dataclass
class Transform:
    position: np.ndarray # global
    rotation: Quaternion # local

@dataclass
class BoneInformation:
    transform: Transform
    is_visible: bool

@dataclass
class Skeleton:
    # TODO: enter Auto-Rig pro export Bone hirarchy
    # HipL: BoneInformation
    # HipR: BoneInformation
    pass

@dataclass
class AnimationState:
    timestamp: int
    skeleton: Skeleton

@dataclass
class RawAnimationSequence:
    states: list[AnimationState]

# ---------------------------------------------------------------------------
'''
FUNCTIONS

'''
# TODO: specify (i think its for COCO / Halpe or both)
def format_bodyparts_kpts(kpts):
    return {
        "head": kpts[0],
        "left_shoulder": kpts[1],
        "right_shoulder": kpts[2],
        "left_elbow": kpts[3],
        "right_elbow": kpts[4],
        "left_hand": kpts[5],
        "right_hand": kpts[6],
        "left_hip": kpts[7],
        "right_hip": kpts[8],
        "left_knee": kpts[9],
        "right_knee": kpts[10],
        "left_foot": kpts[11],
        "right_foot": kpts[12],
    }

def calibrate_global_scene_root():
    # from frames ?
    # or from AnimationSequence ?
    pass # TODO

