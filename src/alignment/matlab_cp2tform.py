"""
Similarity transform like https://www.mathworks.com/help/images/ref/cp2tform.html
"""
from typing import Tuple

import numpy as np
from numpy.linalg import inv, norm, lstsq, matrix_rank as rank


def transform_forward(transformation_matrix: np.ndarray, src_points: np.ndarray) -> np.ndarray:
    """
    Apply forward transformation for `src_points` with `transformation_matrix`
    :param transformation_matrix: matrix for apply transformation from `src_points` to `dst_points`
    :param src_points: source points in 2xK format
    :return: destination points
    """
    src_points = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    dst_points = np.dot(src_points, transformation_matrix)
    dst_points = dst_points[:, 0:-1]
    return dst_points


def transform_inverse(transformation_matrix: np.ndarray, src_points: np.ndarray) -> np.ndarray:
    """
    Apply inverse transformation for `src_points` with `transformation_matrix`
    :param transformation_matrix: matrix for apply transformation from `src_points` to `dst_points`
    :param src_points: source points in 2xK format
    :return: destination points
    """
    transformation_matrix_inverse = inv(transformation_matrix)
    dst_points = transform_forward(transformation_matrix_inverse, src_points)
    return dst_points


def find_nonreflective_similarity(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find non reflective similarity from `src_points` to `dst_points`
    :param src_points: source points in 2xK format
    :param dst_points: destination points in 2xK format
    :return: transformation matrix from `src_points` to `dst_points`;
     inverse transformation matrix, transformation matrix from `dst_points` to `src_points`
    """
    dst_len = dst_points.shape[0]
    dst_x = dst_points[:, 0].reshape(-1, 1)  # use reshape to keep a column vector
    dst_y = dst_points[:, 1].reshape(-1, 1)  # use reshape to keep a column vector

    tmp1 = np.hstack((dst_x, dst_y, np.ones((dst_len, 1)), np.zeros((dst_len, 1))))
    tmp2 = np.hstack((dst_y, -dst_x, np.zeros((dst_len, 1)), np.ones((dst_len, 1))))
    X = np.vstack((tmp1, tmp2))

    src_x = src_points[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    src_y = src_points[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((src_x, src_y))

    if rank(X) >= 4:
        solution, _, _, _ = lstsq(X, U, rcond=-1)
        solution = np.squeeze(solution)
    else:
        raise Exception('Two unique points required.')

    transform_matrix_inverse = np.array([
        [solution[0], -solution[1], 0],
        [solution[1],  solution[0], 0],
        [solution[2],  solution[3], 1]
    ])

    transform_matrix = inv(transform_matrix_inverse)

    transform_matrix[:, 2] = np.array([0, 0, 1])

    return transform_matrix, transform_matrix_inverse


def find_similarity(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find reflective similarity from `src_points` to `dst_points`
    :param src_points: source points in 2xK format
    :param dst_points: destination points in 2xK format
    :return: transformation matrix from `src_points` to `dst_points`;
     inverse transformation matrix, transformation matrix from `dst_points` to `src_points`
    """
    transform_matrix1, transform_matrix_inverse1 = find_nonreflective_similarity(src_points, dst_points)

    dst_points_reflect = dst_points.copy()
    dst_points_reflect[:, 0] = -1 * dst_points_reflect[:, 0]

    transform_matrix2_reflect, _ = find_nonreflective_similarity(src_points, dst_points_reflect)

    # manually reflect the `transform_matrix2_reflect` to undo the reflection done on `dst_points_reflect`
    transform_reflect_y = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    transform_matrix2 = np.dot(transform_matrix2_reflect, transform_reflect_y)

    # Figure out if transform_matrix1 or transform_matrix2 is better
    norm1 = norm(transform_forward(transform_matrix1, src_points) - dst_points)
    norm2 = norm(transform_forward(transform_matrix2, src_points) - dst_points)

    if norm1 <= norm2:
        return transform_matrix1, transform_matrix_inverse1
    else:
        transform_matrix_inverse2 = inv(transform_matrix2)
        return transform_matrix2, transform_matrix_inverse2


def get_similarity_transform_matrix(src_points: np.ndarray, dst_points: np.ndarray, reflective: bool = True)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Get similarity transform matrix, like here https://www.mathworks.com/help/images/ref/cp2tform.html
    :param src_points: source points in 2xK format
    :param dst_points: destination points in 2xK format
    :param reflective: use reflective or non-reflective similarity transform
    :return: transformation matrix from `src_points` to `dst_points`;
     inverse transformation matrix, transformation matrix from `dst_points` to `src_points`
    """
    if reflective:
        transformation_matrix, transformation_matrix_inverse = find_similarity(src_points, dst_points)
    else:
        transformation_matrix, transformation_matrix_inverse = find_nonreflective_similarity(src_points, dst_points)

    # Convert for cv2.warpAffine
    transformation_matrix = transformation_matrix[:, :2].T
    transformation_matrix_inverse = transformation_matrix_inverse[:, :2].T

    return transformation_matrix, transformation_matrix_inverse
