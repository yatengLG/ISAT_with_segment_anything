# -*- coding: utf-8 -*-
# @Author  : LG

from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np


class SegAny:
    def __init__(self, checkpoint):
        if 'vit_b' in checkpoint:
            self.model_type = "vit_b"
        elif 'vit_l' in checkpoint:
            self.model_type = "vit_l"
        elif 'vit_h' in checkpoint:
            self.model_type = "vit_h"
        else:
            raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.image = None

    def set_image(self, image):

        self.predictor.set_image(image)

    def reset_image(self):
        self.predictor.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict(self, input_point, input_label):
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        torch.cuda.empty_cache()
        return masks
