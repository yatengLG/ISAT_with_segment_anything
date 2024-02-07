# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2

import onnxruntime
from typing import Optional, Tuple

from ..utils.transforms import ResizeLongestSide


class SamPredictorONNX:
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    img_size = 1024
    pixel_mean = np.array([123.675, 116.28, 103.53])[None, :, None, None]
    pixel_std = np.array([58.395, 57.12, 57.375])[None, :, None, None]

    def __init__(
            self,
            encoder_path: str,
            decoder_path: str
    ) -> None:
        super().__init__()
        self.encoder = onnxruntime.InferenceSession(encoder_path)
        self.decoder = onnxruntime.InferenceSession(decoder_path)

        # Set the execution provider to GPU if available
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            self.encoder.set_providers(['CUDAExecutionProvider'])
            self.decoder.set_providers(['CUDAExecutionProvider'])

        self.transform = ResizeLongestSide(self.img_size)
        self.reset_image()

    def set_image(
            self,
            image: np.ndarray,
            image_format: str = "RGB",
    ) -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image = input_image.transpose(2, 0, 1)[None, :, :, :]
        self.reset_image()
        self.original_size = image.shape[:2]
        self.input_size = tuple(input_image.shape[-2:])
        input_image = self.preprocess(input_image).astype(np.float32)
        outputs = self.encoder.run(None, {'image': input_image})
        self.features = outputs[0]
        self.is_image_set = True

    def predict(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        point_coords = self.transform.apply_coords(point_coords, self.original_size)
        outputs = self.decoder.run(None, {
            'image_embeddings': self.features,
            'point_coords': point_coords.astype(np.float32),
            'point_labels': point_labels.astype(np.float32)
        })
        scores, low_res_masks = outputs[0], outputs[1]
        masks = self.postprocess_masks(low_res_masks)
        masks = masks > self.mask_threshold

        return masks, scores, low_res_masks

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    def preprocess(self, x: np.ndarray):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = np.pad(x, ((0, 0), (0, 0), (0, padh), (0, padw)), mode='constant', constant_values=0)
        return x

    def postprocess_masks(self, mask: np.ndarray):
        mask = mask.squeeze(0).transpose(1, 2, 0)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = mask[:self.input_size[0], :self.input_size[1], :]
        mask = cv2.resize(mask, (self.original_size[1], self.original_size[0]), interpolation=cv2.INTER_LINEAR)
        mask = mask.transpose(2, 0, 1)[None, :, :, :]
        return mask
