# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
from copy import deepcopy
import cv2
from .utils.transforms_med2d import Med2dTransforms

class Predictor:
    def __init__(self, sam_model):

        super().__init__()
        self.model = sam_model
        self.devices = sam_model.device
        self.transform = Med2dTransforms(256)
        self.reset_image()

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ["RGB", "BGR", ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
            self,
            transformed_image: torch.Tensor,
            original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
                len(transformed_image.shape) == 4
                and transformed_image.shape[1] == 3
                and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            box: Optional[np.ndarray] = None,
            mask_input: Optional[np.ndarray] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."

            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )
        iou_predictions = iou_predictions.to(torch.float)
        low_res_masks = low_res_masks.to(torch.float)
        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    @torch.no_grad()
    def predict_torch(
            self,
            point_coords: Optional[torch.Tensor],
            point_labels: Optional[torch.Tensor],
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        if boxes is not None and boxes.shape[0] > 1:
            mask_list = []
            # Embed prompts
            for i in range(boxes.shape[0]):
                pre_boxes = boxes[i:i + 1, ...]

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=pre_boxes,
                    masks=mask_input,
                )

                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=self.features,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                if multimask_output:
                    max_values, max_indexs = torch.max(iou_predictions, dim=1)
                    max_values = max_values.unsqueeze(1)
                    iou_predictions = max_values
                    low_res_masks = low_res_masks[:, max_indexs]

                # Upscale the masks to the original image resolution
                pre_masks = self.postprocess_masks(low_res_masks, self.model.image_encoder.img_size, self.original_size)

                mask_list.append(pre_masks)
            masks = torch.cat(mask_list, dim=0)

        else:
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_input,
            )

            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            if multimask_output:
                max_values, max_indexs = torch.max(iou_predictions, dim=1)
                max_values = max_values.unsqueeze(1)
                iou_predictions = max_values
                low_res_masks = low_res_masks[:, max_indexs]

            # Upscale the masks to the original image resolution
            masks = self.postprocess_masks(low_res_masks, self.model.image_encoder.img_size, self.original_size)

        if not return_logits:
            sigmoid_output = torch.sigmoid(masks)
            masks = (sigmoid_output > 0.5).float()

        return masks, iou_predictions, low_res_masks

    def postprocess_masks(self, low_res_masks, image_size, original_size):
        ori_h, ori_w = original_size
        masks = F.interpolate(low_res_masks, (image_size, image_size), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

    def apply_coords_torch(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes, original_size, new_size):
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
