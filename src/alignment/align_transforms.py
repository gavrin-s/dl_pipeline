"""
Align transform
"""
from typing import Tuple

import numpy as np
import cv2

from .matlab_cp2tform import get_similarity_transform_matrix


def align_image(src_image: np.ndarray,
                src_points: np.ndarray,
                dst_points: np.ndarray,
                crop_size: Tuple = (224, 224),
                align_type: str = 'similarity') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate and apply affine transform for `src_image`
    :param src_image: source image
    :param src_points: source coordinates, in Kx2 or 2xK format
    :param dst_points: destination coordinates, in Kx2 or 2xK format
    :param crop_size: output image size
    :param align_type: transformation type, for `get_transform_matrix`
    :return: transformed image with size = `crop_size`, transformation matrix
    """
    def _check_points(points: np.ndarray) -> np.ndarray:
        """
        Auxiliary function for check coordinates
        :param points: coordinates in format Kx2 or 2xK
        :return: if the check was successful, coordinates in format 2xK
        """
        points = np.float32(points)
        points_shape = points.shape
        if max(points.shape) < 3 or min(points_shape) != 2:
            raise Exception('points.shape must be (K,2) or (2,K) and K>2')

        if points_shape[0] == 2:
            points = points.T
        return points

    dst_points = _check_points(dst_points)
    src_points = _check_points(src_points)

    if src_points.shape != dst_points.shape:
        raise Exception('src_points and dst_points must have the same shape')

    transformation_matrix = get_transform_matrix(src_points, dst_points, align_type)
    transformed_image = cv2.warpAffine(src_image, transformation_matrix, crop_size)
    return transformed_image, transformation_matrix


def get_transform_matrix(src_points: np.ndarray, dst_points: np.ndarray, align_type: str = 'similarity') -> np.ndarray:
    """

    :param src_points: source coordinates, in 2xK format
    :param dst_points: destination coordinates, in 2xK format
    :param align_type: transformation type, one of:
        1) 'similarity': use similarity transform
        2) 'cv2_affine': use the first 3 points to do affine transform,
                by calling cv2.getAffineTransform()
        3) 'affine': use all points to do affine transform
    :return: transformation matrix
    """
    if align_type == 'cv2_affine':
        return cv2.getAffineTransform(src_points[0:3], dst_points[0:3])
    elif align_type == 'affine':
        return get_affine_transform_matrix(src_points, dst_points)
    else:
        return get_similarity_transform_matrix(src_points, dst_points)[0]


def get_affine_transform_matrix(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Get affine transform matrix 'transformation_matrix' from src_points to dst_points
    :param src_points: source coordinates, in 2xK format
    :param dst_points: destination coordinates, in 2xK format
    :return: affine transform matrix from src_points to dst_points
    """
    n_points = src_points.shape[0]
    ones = np.ones((n_points, 1), src_points.dtype)
    src_points_ = np.hstack([src_points, ones])
    dst_points_ = np.hstack([dst_points, ones])

    solution, _, rank, _ = np.linalg.lstsq(src_points_, dst_points_)
    transformation_matrix = solution[:, :2].T.astype(float)

    return transformation_matrix
