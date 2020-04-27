from typing import Dict
import random
import numpy as np
import cv2
from albumentations import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox
import albumentations.augmentations.functional as F


class KeyPointNormalize(DualTransform):
    def __init__(self, height: int, width: int, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_y = 1.0 / height
        self.scale_x = 1.0 / width

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        new_keypoint = F.keypoint_scale(keypoint, scale_x=self.scale_x, scale_y=self.scale_y)
        return new_keypoint

    def get_transform_init_args_names(self):
        return "height", "width"


class RandomMAxSquaredCrop(DualTransform):
    """Crop a random squared part of the input. Size cropped image equal minimum size of image.

    Args:
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0):
        super(RandomMAxSquaredCrop, self).__init__(always_apply, p)

    def apply(self, img, h_start=0, w_start=0, **params):
        return F.random_crop(img, params["crop_height"], params["crop_width"], h_start, w_start)

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_random_crop(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_random_crop(keypoint, **params)

    def update_params(self, params, **kwargs):
        params = super(RandomMAxSquaredCrop, self).update_params(params, **kwargs)
        min_size = min(params["cols"], params["rows"])
        params["crop_height"] = min_size
        params["crop_width"] = min_size
        return params


class PadToMaxSize(DualTransform):
    """Pad to max side.

    Args:
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, lisft of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    lisft of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]
        max_size = max(rows, cols)

        if rows < max_size:
            h_pad_top = int((max_size - rows) / 2.0)
            h_pad_bottom = max_size - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < max_size:
            w_pad_left = int((max_size - cols) / 2.0)
            w_pad_right = max_size - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.value
        )

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.mask_value
        )

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return ("border_mode", "value", "mask_value")


class TransposeMask:
    """
    For segmentation-models-pytorch.
    """
    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        kwargs["mask"] = kwargs["mask"].transpose((2, 0, 1)).astype('float32')
        return kwargs
