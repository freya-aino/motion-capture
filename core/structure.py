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
class CMUPanopticDatapoint:
    pass # TODO

@dataclass
class Human36MDatapoint:
    pass # TODO

@dataclass
class COCOFullBodyDatapoint:
    pass # TODO

@dataclass
class Transform:
    position: np.ndarray # global
    rotation: Quaternion # local
    position_dt: np.ndarray # global
    rotation_dt: Quaternion # local

    def __init__(self):
        self.position = np.zeros((3), dtype=np.float32)
        self.rotation = Quaternion()
        self.position_dt = np.zeros((3), dtype=np.float32)
        self.rotation_dt = Quaternion()
        return self

@dataclass
class BoneInformation:
    transform: Transform
    is_visible: bool

@dataclass
class AutoRigProBoneSkeleton:
    # TODO: enter Auto-Rig pro export Bone hirarchy
    # HipL: BoneInformation
    # HipR: BoneInformation

    def import_CMUPanoptic(self, cmp_panoptic: CMUPanopticDatapoint):
        pass # TODO
    def import_Human36M(self, human36m: Human36MDatapoint):
        pass # TODO

    def get_skeleton_root(self):
        pass # TODO

@dataclass
class AnimationState:
    timestamp: int
    skeleton: AutoRigProBoneSkeleton

@dataclass
class RawAnimationSequence:
    states: list[AnimationState]

# ---------------------------------------------------------------------------
'''
FUNCTIONS

'''

def calibrate_global_scene_root():
    # from frames ?
    # or from AnimationSequence ?
    pass # TODO

