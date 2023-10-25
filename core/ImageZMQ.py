import json
import threading
import traceback
import numpy as np

from numpy import ndarray
from dataclasses import dataclass
from datetime import datetime
from imagezmq import ImageSender, ImageHub


@dataclass
class CameraFramePacket:
    camera_id: str
    timestamp: datetime
    frame: ndarray


class ImageZMQVideoStreamSender:
    '''
        This modle is for the purpose of sending images over ImageZMQ
        
        its currenlty only configured for pub sub
    '''

    def __init__(self, hostname:str, port:str):
        
        self.address = f"tcp://{hostname}:{port}"
        self.image_sender = ImageSender(connect_to=self.address, REQ_REP=False)

    def send(self, camera_frame_packet: CameraFramePacket) -> None:
        information = {
            "camera_id": camera_frame_packet.camera_id,
            "timestamp": camera_frame_packet.timestamp
        }
        self.image_sender.send_image(json.dumps(information), camera_frame_packet.frame)


class ImageZMQVideoStreamSubscriber:
    '''
        This modules purpose is to read the currently active frame asyncronously from the message queue.

        For task that take longer than the fps of the camera to compute the ZMQ queue would fill up the RAM otherwise,
        this discards the frames inbetween

    '''

    #TODO: write this with asyncIO instead of threading (if asyncIO is even supported for this)

    def __init__(self, hostname:str, port:str):
        self.hostname = hostname
        self.port = port
        self.stop = False
        self.data_ready = threading.Event()
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

        self.current_packet = CameraFramePacket(
            camera_id="N/A",
            timestamp=0,
            frame=np.zeros([0]),
        )

    def receive(self, timeout:float = 15.0) -> CameraFramePacket:
        flag = self.data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(f"Timeout while reading from subscriber tcp://{self.hostname}:{self.port}")
        self.data_ready.clear()
        return self.current_packet

    def run(self) -> None:
        receiver = ImageHub(f"tcp://{self.hostname}:{self.port}", REQ_REP=False)

        try:
            while not self.stop:
                info, image = receiver.recv_image()
                info = json.loads(info)
                
                self.current_packet = CameraFramePacket(
                    camera_id=info["camera_id"],
                    timestamp=info["timestamp"],
                    frame=image,
                )

                self.data_ready.set()
            receiver.close()
        except:
            traceback.print_exc()
        finally:
            receiver.close()

    def close(self) -> None:
        self.stop = True
        self.data_ready.set() # this is required to not have it stuck on data_ready.wait()
