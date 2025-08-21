# -*- coding: utf-8 -*-
# @Author  : LG
from typing import Union

import torch
import numpy as np
import platform
from PIL import Image
from collections import OrderedDict
import os
from ISAT.segment_any.sam2.utils.misc import AsyncVideoFrameLoader

osplatform = platform.system()


class SegAny:
    """
    Segment Anything model class.

    Arguments:
        checkpoint: checkpoint path.
        use_bfloat16: model dtype is bfloat16. default is True.

    Attributes:
        checkpoint (str): checkpoint path.
        model_dtype (torch.dtype): model dtype. eg: torch.bfloat16, torch.float32
        model_source (str): model source. eg: mobile_sam, sam_hq, sam2, sam2.1, ...
        model_type (str): model type. eg: vit_b, vit_t, hiera_small, ...
        predictor (SamPredictor): Sam Predictor.
    """
    def __init__(self, checkpoint:str, use_bfloat16:bool=True):
        print('--' * 20)
        print('* Init SAM... *')
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
                model_type = "hiera_tiny"
            elif 'hiera_small' in checkpoint:
                model_type = "hiera_small"
            elif 'hiera_base_plus' in checkpoint:
                model_type = 'hiera_base_plus'
            elif 'hiera_large' in checkpoint:
                model_type = 'hiera_large'
            else:
                raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))
            # sam2.1 or sam2
            model_source = 'sam2.1' if 'sam2.1' in checkpoint else 'sam2'
            self.model_type = "{}_{}".format(model_source, model_type)
            self.model_source = model_source

        elif 'med2d' in checkpoint:
            from ISAT.segment_any.segment_anything_med2d import sam_model_registry
            from ISAT.segment_any.segment_anything_med2d.predictor_for_isat import Predictor as SamPredictor
            self.model_type = "vit_b"
            self.model_source = 'sam_med2d'

        torch.cuda.empty_cache()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('  - device  : {}'.format(self.device))
        print('  - dtype   : {}'.format(self.model_dtype))
        print('  - loading : {}'.format(checkpoint))
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)

        sam = sam.eval().to(self.model_dtype)

        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print('* Init SAM finished *')
        print('--'*20)
        self.image = None

    def set_image(self, image: np.ndarray) -> None:
        """
        Set image to segment.

        Arguments:
            image (np.ndarray): The image to segment.
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype, enabled=torch.cuda.is_available()):
            self.image = image
            self.predictor.set_image(image)

    def reset_image(self) -> None:
        """
        Reset image to segment.
        """
        self.predictor.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict_with_point_prompt(self, input_point: Union[list, np.ndarray], input_label: Union[list, np.ndarray]) -> torch.Tensor:
        """
        Segment with point prompt.

        Arguments:
            input_point (list | np.ndarray): The input points. [(x1, y1), (x2, y2), ...]
            input_label (list | np.ndarray): The label of points, support value 0 or 1. [0, 0, ...]
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype, enabled=torch.cuda.is_available()):

            if 'sam2' not in self.model_type:
                input_point = np.array(input_point)
                input_label = np.array(input_label)
            else:
                input_point = input_point
                input_label = input_label

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            if self.model_source == 'sam_med2d':
                return masks

            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            torch.cuda.empty_cache()
            return masks

    def predict_with_box_prompt(self, box: Union[list, np.ndarray]) -> torch.Tensor:
        """
        Segment with box prompt.

        Arguments:
            box (list | np.ndarray): [xmin, ymin, xmax, ymax]
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype, enabled=torch.cuda.is_available()):
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            torch.cuda.empty_cache()
            return masks


class SegAnyVideo:
    """
    Segment Anything model class for video segmentation. Only SAM2 or SAM2.1 model.

    Arguments:
        checkpoint: checkpoint path.
        use_bfloat16: model dtype is bfloat16. default is True.

    Attributes:
        checkpoint (str): checkpoint path.
        model_dtype (torch.dtype): model dtype. eg: torch.bfloat16, torch.float32
        model_source (str): model source. sam2 or sam2.1 .
        model_type (str): model type. eg: hiera_tiny, hiera_small, hiera_base_plus or hiera_large.
        predictor (SamPredictor): Sam Predictor.
    """
    def __init__(self, checkpoint: str, use_bfloat16: bool = True):
        print('--'*20)
        print('* Init SAM for video... *')

        self.checkpoint = checkpoint
        self.model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.model_source = None

        self.inference_state = {}

        if 'sam2' in checkpoint:
            from ISAT.segment_any.sam2.build_sam import sam_model_registry
            # sam2
            if 'hiera_tiny' in checkpoint:
                model_type = "hiera_tiny"
            elif 'hiera_small' in checkpoint:
                model_type = "hiera_small"
            elif 'hiera_base_plus' in checkpoint:
                model_type = 'hiera_base_plus'
            elif 'hiera_large' in checkpoint:
                model_type = 'hiera_large'
            else:
                raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))

            # sam2.1 or sam2
            model_source = 'sam2.1' if 'sam2.1' in checkpoint else 'sam2'
            self.model_type = "{}_{}_video".format(model_source, model_type)
            self.model_source = model_source

        torch.cuda.empty_cache()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('  - device  : {}'.format(self.device))
        print('  - dtype   : {}'.format(self.model_dtype))
        print('  - loading : {}'.format(checkpoint))
        self.predictor = sam_model_registry[self.model_type](checkpoint=checkpoint)
        self.predictor = self.predictor.eval().to(self.model_dtype)
        self.predictor.to(device=self.device)
        print('* Init SAM for video finished *')
        print('--'*20)

    def init_state(
            self,
            image_root: str,
            image_name_list: list,
            offload_video_to_cpu: bool=True,
            offload_state_to_cpu: bool=True,
            async_loading_frames: bool=False,
    ) -> None:
        """
        Init state for video segmentation. Need read all images.

        Arguments:
            image_root (str): The directory of images.
            image_name_list (list): list of image file names to segment.
            offload_video_to_cpu (bool): offload frames to CPU.
            offload_state_to_cpu (bool): offload state to CPU.
            async_loading_frames (bool): asynchronize loading frames.
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype, enabled=torch.cuda.is_available()):

            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)

            num_frames = len(image_name_list)
            img_paths = [os.path.join(image_root, frame_name) for frame_name in image_name_list]
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

            image_size = self.predictor.image_size

            if async_loading_frames:
                lazy_images = AsyncVideoFrameLoader(
                    img_paths, image_size, offload_video_to_cpu, img_mean, img_std
                )
                # return lazy_images, lazy_images.video_height, lazy_images.video_width
                images, video_height, video_width = lazy_images, lazy_images.video_height, lazy_images.video_width
            else:
                images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
                for n, img_path in enumerate(img_paths):
                    images[n], video_height, video_width = self._load_img_as_tensor(img_path, image_size)
                if not offload_video_to_cpu:
                    images = images.cuda()
                    img_mean = img_mean.cuda()
                    img_std = img_std.cuda()

                images -= img_mean
                images /= img_std
                # images, video_height, video_width = load_video_frames

            inference_state = {}
            inference_state["images"] = images
            inference_state["num_frames"] = len(images)
            # whether to offload the video frames to CPU memory
            # turning on this option saves the GPU memory with only a very small overhead
            inference_state["offload_video_to_cpu"] = offload_video_to_cpu
            # whether to offload the inference state to CPU memory
            # turning on this option saves the GPU memory at the cost of a lower tracking fps
            # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
            # and from 24 to 21 when tracking two objects)
            inference_state["offload_state_to_cpu"] = offload_state_to_cpu
            # the original video height and width, used for resizing final output scores
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
            inference_state["device"] = torch.device("cuda")
            if offload_state_to_cpu:
                inference_state["storage_device"] = torch.device("cpu")
            else:
                inference_state["storage_device"] = torch.device("cuda")
            # inputs on each frame
            inference_state["point_inputs_per_obj"] = {}
            inference_state["mask_inputs_per_obj"] = {}
            # visual features on a small number of recently visited frames for quick interactions
            inference_state["cached_features"] = {}
            # values that don't change across frames (so we only need to hold one copy of them)
            inference_state["constants"] = {}
            # mapping between client-side object id and model-side object index
            inference_state["obj_id_to_idx"] = OrderedDict()
            inference_state["obj_idx_to_id"] = OrderedDict()
            inference_state["obj_ids"] = []
            # A storage to hold the model's tracking results and states on each frame
            inference_state["output_dict"] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
            inference_state["output_dict_per_obj"] = {}
            # A temporary storage to hold new outputs when user interact with a frame
            # to add clicks or mask (it's merged into "output_dict" before propagation starts)
            inference_state["temp_output_dict_per_obj"] = {}
            # Frames that already holds consolidated outputs from click or mask inputs
            # (we directly use their consolidated outputs during tracking)
            inference_state["consolidated_frame_inds"] = {
                "cond_frame_outputs": set(),  # set containing frame indices
                "non_cond_frame_outputs": set(),  # set containing frame indices
            }
            # metadata for each tracking frame (e.g. which direction it's tracked)
            inference_state["tracking_has_started"] = False
            inference_state["frames_already_tracked"] = {}
            # Warm up the visual backbone and cache the image feature on frame 0
            self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

            self.inference_state = inference_state
            self.reset_state()

            print('init state finished.')

    def reset_state(self) -> None:
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(self.inference_state)
        # Remove all object ids
        self.inference_state["obj_id_to_idx"].clear()
        self.inference_state["obj_idx_to_id"].clear()
        self.inference_state["obj_ids"].clear()
        self.inference_state["point_inputs_per_obj"].clear()
        self.inference_state["mask_inputs_per_obj"].clear()
        self.inference_state["output_dict_per_obj"].clear()
        self.inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    @staticmethod
    def _load_img_as_tensor(img_path, image_size):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size  # the original video size
        return img, video_height, video_width

    def add_new_mask(self, frame_idx: int, ann_obj_id: int, mask: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Add new mask prompt for video segmentation.

        Arguments:
            frame_idx (int): Frame index.
            ann_obj_id (int): Object ID.
            mask (torch.Tensor | np.ndarray): New mask prompt.
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype, enabled=torch.cuda.is_available()):
            self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                mask=mask
            )
