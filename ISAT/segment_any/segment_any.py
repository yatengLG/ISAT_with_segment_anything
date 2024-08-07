# -*- coding: utf-8 -*-
# @Author  : LG


import torch
import numpy as np
import timm
import platform

osplatform = platform.system()

class SegAny:
    def __init__(self, checkpoint:str, use_bfloat16:bool=True):
        self.checkpoint = checkpoint
        self.model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.model_source = None
        if 'mobile_sam' in checkpoint:
            # mobile sam
            from ISAT.segment_any.mobile_sam import sam_model_registry, SamPredictor
            self.model_type = "vit_t"
            self.model_source = 'mobile_sam'
        elif 'edge_sam' in checkpoint:
            # edge_sam
            from ISAT.segment_any.edge_sam import sam_model_registry, SamPredictor
            self.model_type = "edge_sam"
            self.model_source = 'edge_sam'
        elif 'sam_hq_vit' in checkpoint:
            # sam hq
            from ISAT.segment_any.segment_anything_hq import sam_model_registry, SamPredictor
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
            if torch.__version__ > '2.1.1' and osplatform == 'Linux':
                # 暂时只测试了2.1.1环境下的运行;2.0不确定；1.x不可以
                # 暂时不使用sam-fast
                # from ISAT.segment_anything_fast import sam_model_registry as sam_model_registry
                # from ISAT.segment_anything_fast import SamPredictor
                # print('segment_anything_fast')
                from ISAT.segment_any.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            else:
                # windows下，现只支持 2.2.0+dev，且需要其他依赖；等后续正式版本推出后，再进行支持
                # （如果想提前在windows下试用，可参考https://github.com/pytorch-labs/segment-anything-fast项目进行环境配置）
                from ISAT.segment_any.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            if 'vit_b' in checkpoint:
                self.model_type = "vit_b"
            elif 'vit_l' in checkpoint:
                self.model_type = "vit_l"
            elif 'vit_h' in checkpoint:
                self.model_type = "vit_h"
            else:
                raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))
            self.model_source = 'sam'
        elif 'sam2' in checkpoint:
            from ISAT.segment_any.sam2.build_sam import sam_model_registry
            from ISAT.segment_any.sam2.sam2_image_predictor import SAM2ImagePredictor as SamPredictor
            # sam2
            if 'hiera_tiny' in checkpoint:
                self.model_type = "sam2_hiera_tiny"
            elif 'hiera_small' in checkpoint:
                self.model_type = "sam2_hiera_small"
            elif 'hiera_base_plus' in checkpoint:
                self.model_type = 'sam2_hiera_base_plus'
            elif 'hiera_large' in checkpoint:
                self.model_type = 'sam2_hiera_large'
            else:
                raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))
            self.model_source = 'sam2'
            # sam2 在float32下运行时存在报错，暂时只在bfloat16下运行
            # self.model_dtype = torch.bfloat16

        elif 'med2d' in checkpoint:
            from ISAT.segment_any.segment_anything_med2d import sam_model_registry
            from ISAT.segment_any.segment_anything_med2d.predictor_for_isat import Predictor as SamPredictor
            self.model_type = "vit_b"
            self.model_source = 'sam_med2d'

        torch.cuda.empty_cache()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('- device  : {}'.format(self.device))
        print('- dtype   : {}'.format(self.model_dtype))
        print('- loading : {}'.format(checkpoint))
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)

        sam = sam.eval().to(self.model_dtype)

        sam.to(device=self.device)
        self.predictor_with_point_prompt = SamPredictor(sam)
        print('- loaded')
        self.image = None

    def set_image(self, image):
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype):
            self.image = image
            self.predictor_with_point_prompt.set_image(image)

    def reset_image(self):
        self.predictor_with_point_prompt.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict_with_point_prompt(self, input_point, input_label):
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype):

            if 'sam2' not in self.model_type:
                input_point = np.array(input_point)
                input_label = np.array(input_label)
            else:
                input_point = input_point
                input_label = input_label

            masks, scores, logits = self.predictor_with_point_prompt.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            if self.model_source == 'sam_med2d':
                return masks

            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, _, _ = self.predictor_with_point_prompt.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            torch.cuda.empty_cache()
            return masks

    def predict_with_box_prompt(self, box):
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype):
            masks, scores, logits = self.predictor_with_point_prompt.predict(
                box=box,
                multimask_output=False,
            )
            torch.cuda.empty_cache()
            return masks
