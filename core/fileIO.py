import cv2
import numpy as np
import shutil
import os
import glob

from logging import getLogger
from traceback import format_exc

# -----------------------------------------------------------------------------------------------------------

logger = getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------


def remove_file(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

def empty_directory(path):
    for i in glob.glob(os.path.join(path, "*")):
        remove_file(i)

def load_all_frames_from_video(video_path: str):
    
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = np.zeros((num_frames, h, w, 3), dtype = np.uint8)
    
    try:
        
        logger.info(f"loading {num_frames} frames from {video_path}")
        
        for ii in range(num_frames):
            
            ret, frame = cap.read()
            assert ret, "VideoCapture is empty before end of sequence"
            
            out[ii] = frame[:]
            
    except:
        logger.error(f"error while loading frames: {format_exc()}")
    finally:
        cap.release()
    return out
