# -*- coding: utf-8 -*-
# @Author  : LG


from .transforms import ResizeLongestSide
import numpy as np
from typing import Tuple
import cv2


class Med2dTransforms(ResizeLongestSide):

    def apply_image(self, image: np.ndarray):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        return image

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        return (long_side_length, long_side_length)
