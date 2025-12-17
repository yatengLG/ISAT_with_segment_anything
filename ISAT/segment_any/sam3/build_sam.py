# -*- coding: utf-8 -*-
# @Author  : LG

from .model_builder import build_sam3_image_model, build_sam3_video_model
from .model.sam3_image_processor import Sam3Processor
from typing import Optional
import os

import torch
import numpy as np
from torchvision.transforms import v2
import PIL


bpe_path = os.path.join(__file__, "..", "bpe_simple_vocab_16e6.txt.gz")

def build_sam3(checkpoint):
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    return model

def build_sam3_video(checkpoint):
    model = build_sam3_video_model(
        checkpoint_path=checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    return predictor

sam_model_registry = {
    "sam3": build_sam3,
    "sam3_video": build_sam3_video,
}


class Sam3Predictor:
    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self._sam3_processor = Sam3Processor(model, resolution, device, confidence_threshold)

    def _transforms(self, image):
        image = v2.functional.to_image(image).to(self.model.device)
        image = self._sam3_processor.transform(image).unsqueeze(0)
        return image

    @torch.inference_mode()
    def encode(self, image):
        _orig_hw = tuple(image.shape[:2])
        image = self._transforms(image)

        backbone_out = self.model.backbone.forward_image(image)
        sam2_backbone_out = backbone_out["sam2_backbone_out"]
        sam2_backbone_out["backbone_fpn"][0] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        )
        _, vision_feats, _, _ = (
            self.model.inst_interactive_predictor.model._prepare_backbone_features(
                backbone_out["sam2_backbone_out"]
            )
        )
        vision_feats[-1] = (
                vision_feats[-1] + self.model.inst_interactive_predictor.model.no_mem_embed
        )
        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(
                vision_feats[::-1],
                self.model.inst_interactive_predictor._bb_feat_sizes[::-1]
            )
                ][::-1]
        _features = {
            "image_embed": feats[-1],
            "high_res_feats": tuple(feats[:-1]),
        }
        return _features, _orig_hw

    def set_image(self, image: np.ndarray):
        _features, _orig_hw = self.encode(image)
        self.model.inst_interactive_predictor._is_image_set = True
        self.model.inst_interactive_predictor._orig_hw = [_orig_hw]
        self.model.inst_interactive_predictor._features = _features

    def reset_image(self):
        self.model.inst_interactive_predictor._is_image_set = True
        self.model.inst_interactive_predictor._orig_hw = None
        self.model.inst_interactive_predictor._features = None

    def predict(self,
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = True,
                return_logits: bool = False,
                normalize_coords=True,
                ):
        masks, scores, logits  = self.model.inst_interactive_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )
        return masks, scores, logits

    def predict_with_text_prompt(self, image: PIL.Image, prompt:str):
        inference_state = self._sam3_processor.set_image(image)
        self._sam3_processor.reset_all_prompts(inference_state)
        inference_state = self._sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = inference_state["masks"]
        masks = masks.squeeze(1).cpu().numpy()
        scores = inference_state["scores"]
        return masks, scores
