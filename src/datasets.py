"""Implementation datasets"""
import os
import warnings
from typing import Callable, Tuple, Dict, Optional, List, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from src.alignment import align_image


class DatasetBase(Dataset):
    def __init__(self, caching: bool = False,
                 preprocessing: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 postprocessing: Optional[Callable] = None):
        super().__init__()
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.postprocessing = postprocessing
        self.caching = caching

    def set_transforms(self, transforms: Callable):
        """
        Set new transformations without reload dataset.
        """
        self.transforms = transforms

    def set_preprocessing(self, preprocessing: Callable):
        """
        Set new preprocessing without reload dataset
        """
        self.preprocessing = preprocessing

    def set_postprocessing(self, postprocessing: Callable):
        """
        Set new postprocessing without reload dataset
        """
        self.postprocessing = postprocessing

    def _collate_data(self):
        raise NotImplementedError

    def _load_data_from_disk(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class SegmentationDataset(DatasetBase):
    """Dataset for binary segmentation of palm"""
    def __init__(self, main_dict: Dict, data_path: str, caching: bool = False,
                 preprocessing: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 postprocessing: Optional[Callable] = None):
        super().__init__(caching=caching,
                         preprocessing=preprocessing,
                         transforms=transforms,
                         postprocessing=postprocessing)
        self.main_dict = main_dict
        self.data_path = data_path
        self.image_names = tuple(main_dict.keys())

        # for cache all data in RAM
        self.images = np.array([])
        self.masks = np.array([])
        if caching:
            self._collate_data()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.caching:
            data = self._load_data_from_disk(idx)
        else:
            data = dict(image=self.images[idx].copy(), mask=self.masks[idx].copy())

        if self.transforms:
            data = _run_transformation(self.transforms, data)

        if self.postprocessing:
            data = _run_transformation(self.postprocessing, data)

        return data["image"], data["mask"]

    def __len__(self):
        return len(self.image_names)

    def _collate_data(self):
        """
        Collate and save all data
        """
        tmp_images = []
        tmp_masks = []
        for idx in range(self.__len__()):
            data = self._load_data_from_disk(idx)
            tmp_images.append(data["image"])
            tmp_masks.append(data["mask"])
        self.images = np.array(tmp_images)
        self.masks = np.array(tmp_masks)

    def _load_data_from_disk(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load one data from disk by idx.
        """
        image_name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.data_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        contours = np.array(self.main_dict[image_name])
        if len(contours.shape) != 3:
            contours = contours[None, ...]
        contours = contours.reshape((len(contours), -1, 2)).astype(np.int32)
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        mask = cv2.fillPoly(mask, pts=contours, color=(1, 1, 1))  # .astype(np.float32)

        data = dict(image=image, mask=mask)

        if self.preprocessing:
            data = _run_transformation(self.preprocessing, data)

        return data


class KeyPointsDataset(DatasetBase):
    """
    Keypoints detection dataset with the ability to save data in RAM
    """
    def __init__(self, main_dict: Dict, data_path: str, caching: bool = False,
                 preprocessing: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 postprocessing: Optional[Callable] = None):
        """
        :param main_dict: dict in format {"image_name": keypoints}
        :param data_path: folder with images
        :param caching: if True all data will be downloaded and saved in RAM, else every time data is loaded from disk
        :param preprocessing: preliminary processing
        :param transforms: augmentations
        :param postprocessing: transformations for NN
        """
        super().__init__(caching=caching,
                         preprocessing=preprocessing,
                         transforms=transforms,
                         postprocessing=postprocessing)

        self.main_dict = main_dict
        self.data_path = data_path
        self.image_names = tuple(main_dict.keys())

        # for cache all data in RAM
        self.images = np.array([])
        self.keypoints = np.array([])
        if caching:
            self._collate_data()

    def __getitem__(self, idx: int) -> Tuple:
        if not self.caching:
            data = self._load_data_from_disk(idx)
        else:
            data = dict(image=self.images[idx].copy(), keypoints=self.keypoints[idx].copy())

        if self.transforms:
            data = _run_transformation(self.transforms, data)

        if self.postprocessing:
            data = _run_transformation(self.postprocessing, data)
            data["keypoints"] = torch.Tensor(data["keypoints"]).flatten()

        return data["image"], data["keypoints"]

    def __len__(self):
        return len(self.image_names)

    def _load_data_from_disk(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load one data from disk by idx.
        """
        image_name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.data_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = np.array(self.main_dict[image_name])
        data = dict(image=image, keypoints=keypoints)

        if self.preprocessing:
            data = _run_transformation(self.preprocessing, data)
        data["keypoints"] = np.array(data["keypoints"])

        return data

    def _collate_data(self):
        """
        Collate and save all data
        """
        tmp_images = []
        tmp_keypoints = []
        for idx in range(self.__len__()):
            image, keypoints = self._load_data_from_disk(idx)
            tmp_images.append(image)
            tmp_keypoints.append(keypoints)
        self.images = np.array(tmp_images)
        self.keypoints = np.array(tmp_keypoints)


class CustomConcatDataset(ConcatDataset):
    """
    Wrapper for `ConcatDataset` with ability change transformations on the fly.
    """
    def set_transforms(self, transforms: Callable):
        for d in self.datasets:
            d.set_transforms(transforms)


def _run_transformation(transformation: Callable, data: Dict) -> Dict[str, np.ndarray]:
    """
    Run transformation while not success
    :param transformation:
    :param data:
    :return:
    """
    warning_iterations = 10
    counter = 0
    while True:
        counter += 1
        transformed_data = transformation(**data.copy())

        # keypoints validation
        if "keypoints" in data:
            transformed_data["keypoints"] = np.array(transformed_data["keypoints"])
            if len(data["keypoints"]) != len(transformed_data["keypoints"]):
                if counter >= warning_iterations:
                    warnings.warn("Too many iterations for transformation!")
                    transformed_data = data
                continue
        break
    return transformed_data


class BalancedBatchSampler:
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, labels: np.ndarray, n_classes: int, n_samples: int):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset_length = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self) -> List[int]:
        self.count = 0
        while self.count + self.batch_size < self.dataset_length:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:
                               self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self) -> int:
        return self.dataset_length // self.batch_size
