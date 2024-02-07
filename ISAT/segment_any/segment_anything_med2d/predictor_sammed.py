import numpy as np
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
from copy import deepcopy
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

class SammedPredictor:
    def __init__(self, sam_model):

        super().__init__()
        self.model = sam_model
        self.devices = sam_model.device
        self.reset_image()
        

    def set_image(self,image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ["RGB","BGR",], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        if self.model.pixel_mean.device.type == 'cuda':
            pixel_mean, pixel_std = self.model.pixel_mean.squeeze().cpu().numpy(), self.model.pixel_std.squeeze().cpu().numpy()
            input_image = (image - pixel_mean) / pixel_std
        else:
            pixel_mean, pixel_std = self.model.pixel_mean.squeeze().numpy(), self.model.pixel_std.squeeze().numpy()
            input_image = (image - pixel_mean) / pixel_std

        ori_h, ori_w, _ = input_image.shape
        self.original_size = (ori_h, ori_w)
        self.new_size = (self.model.image_encoder.img_size, self.model.image_encoder.img_size)
        transforms = self.transforms(self.new_size)
        augments = transforms(image=input_image)
        input_image = augments['image'][None, :, :, :]

        assert (
            len(input_image.shape) == 4
            and input_image.shape[1] == 3
            and max(*input_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

        self.features = self.model.image_encoder(input_image.to(self.device))
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
            
            point_coords = self.apply_coords(point_coords, self.original_size, self.new_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            box = self.apply_boxes(box, self.original_size, self.new_size)
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
                pre_boxes = boxes[i:i+1,...]
         
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
        masks = F.interpolate(low_res_masks,(image_size, image_size), mode="bilinear", align_corners=False)
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


    def transforms(self, new_size):
        Transforms = []
        new_h, new_w = new_size
        Transforms.append(A.Resize(int(new_h), int(new_w), interpolation=cv2.INTER_NEAREST))
        Transforms.append(ToTensorV2(p=1.0))
        return A.Compose(Transforms, p=1.)

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
