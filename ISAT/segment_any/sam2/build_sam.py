# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hiera_tiny(checkpoint=None):
    return build_sam2(
        'sam2_hiera_t.yaml',
        ckpt_path=checkpoint,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
    )


def build_sam2_hiera_small(checkpoint=None):
    return build_sam2(
        'sam2_hiera_s.yaml',
        ckpt_path=checkpoint,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
    )


def build_sam2_hiera_base_plus(checkpoint=None):
    return build_sam2(
        'sam2_hiera_b+.yaml',
        ckpt_path=checkpoint,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
    )


def build_sam2_hiera_large(checkpoint=None):
    return build_sam2(
        'sam2_hiera_l.yaml',
        ckpt_path=checkpoint,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
    )


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=ISAT.segment_any.sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2_video_predictor_hiera_tiny(checkpoint=None):
    return build_sam2_video_predictor(
        'sam2_hiera_t.yaml',
        ckpt_path=checkpoint,
    )


def build_sam2_video_predictor_hiera_small(checkpoint=None):
    return build_sam2_video_predictor(
        'sam2_hiera_s.yaml',
        ckpt_path=checkpoint,
    )


def build_sam2_video_predictor_hiera_base_plus(checkpoint=None):
    return build_sam2_video_predictor(
        'sam2_hiera_b+.yaml',
        ckpt_path=checkpoint,
    )


def build_sam2_video_predictor_hiera_large(checkpoint=None):
    return build_sam2_video_predictor(
        'sam2_hiera_l.yaml',
        ckpt_path=checkpoint,
    )


sam_model_registry = {
    "sam2_hiera_tiny": build_sam2_hiera_tiny,
    "sam2_hiera_small": build_sam2_hiera_small,
    "sam2_hiera_base_plus": build_sam2_hiera_base_plus,
    "sam2_hiera_large": build_sam2_hiera_large,

    "sam2_hiera_video_tiny": build_sam2_video_predictor_hiera_tiny,
    "sam2_hiera_video_small": build_sam2_video_predictor_hiera_small,
    "sam2_hiera_video_base_plus": build_sam2_video_predictor_hiera_base_plus,
    "sam2_hiera_video_large": build_sam2_video_predictor_hiera_large,
}