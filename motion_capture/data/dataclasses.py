# from abc import abstractmethod
# from typing import Union
# from torch import tensor, int16
# from dataclasses import dataclass


# @dataclass
# class ImagePoint:
#     x: int
#     y: int
    
#     @abstractmethod
#     def as_tensor(self):
#         pass

# @dataclass
# class BoundingBox:
#     x: int # left
#     y: int # up
#     width: int
#     height: int
    
#     def as_tensor(self, stack=False):
#         if stack:
#             return tensor([[self.x, self.y], [self.width, self.height]], dtype=int16)
        
#         return tensor([self.x, self.y, self.width, self.height], dtype=int16)

# @dataclass
# class Keypoints:
#     x: tensor
#     y: tensor
#     z: Union[tensor, None] = None
#     visible: Union[tensor, None] = None
#     score: Union[tensor, None] = None
    