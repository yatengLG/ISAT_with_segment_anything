# -*- coding: utf-8 -*-
# @Author  : LG

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
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
        self.predictor = SamAutomaticMaskGenerator(sam)
        self.predictor_with_point_prompt = SamPredictor(sam)
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

    def predict(self, image):
        self.image = image
        masks = self.predictor.generate(image)
        torch.cuda.empty_cache()
        return masks


if __name__ == '__main__':
    from PIL import Image
    import time
    import matplotlib.pyplot as plt
    time1 = time.time()
    seg = SegAny('sam_vit_h_4b8939.pth')
    image = np.array(Image.open('../example/images/000000000113.jpg'))
    time2 = time.time()
    print(time2-time1)
    # seg.set_image()
    masks = seg.predict(image)
    print(time.time() - time2)
    print(masks)
    for mask in masks:
        mask = mask['segmentation']
        plt.imshow(mask)
        plt.show()
