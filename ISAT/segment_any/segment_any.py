# -*- coding: utf-8 -*-
# @Author  : LG


import torch
import numpy as np
import timm


class SegAny:
    def __init__(self, checkpoint):
        self.model_source = None
        if 'mobile_sam' in checkpoint:
            # mobile sam
            from ISAT.mobile_sam import sam_model_registry, SamPredictor
            self.model_type = "vit_t"
            self.model_source = 'mobile_sam'
        elif 'sam_hq_vit' in checkpoint:
            # sam hq
            from ISAT.segment_anything_hq import sam_model_registry, SamPredictor
            if 'vit_b' in checkpoint:
                self.model_type = "vit_b"
            elif 'vit_l' in checkpoint:
                self.model_type = "vit_l"
            elif 'vit_h' in checkpoint:
                self.model_type = "vit_h"
            elif 'vit_tiny' in checkpoint:
                self.model_type = "vit_tiny"
            else:
                raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))
            self.model_source = 'sam_hq'

        elif 'sam_vit' in checkpoint:
            # sam
            from ISAT.segment_anything import sam_model_registry, SamPredictor
            if 'vit_b' in checkpoint:
                self.model_type = "vit_b"
            elif 'vit_l' in checkpoint:
                self.model_type = "vit_l"
            elif 'vit_h' in checkpoint:
                self.model_type = "vit_h"
            else:
                raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))
            self.model_source = 'sam'

        torch.cuda.empty_cache()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('- device {}'.format(self.device))
        print('- loading {}'.format(checkpoint))
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.predictor_with_point_prompt = SamPredictor(sam)
        print('- loaded')
        self.image = None

    def set_image(self, image):
        self.image = image
        self.predictor_with_point_prompt.set_image(image)

    def reset_image(self):
        self.predictor_with_point_prompt.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict_with_point_prompt(self, input_point, input_label):
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        masks, scores, logits = self.predictor_with_point_prompt.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, _ = self.predictor_with_point_prompt.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        torch.cuda.empty_cache()
        return masks
