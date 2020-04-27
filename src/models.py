from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import pretrainedmodels.models as pretrained_models
from pretrainedmodels import pretrained_settings
import torchvision.models as torch_models
from efficientnet_pytorch import EfficientNet, utils as efficientnet_utils
import segmentation_models_pytorch as smp

PRETRAINED_MODELS = ["resnet50", "resnet18", "resnet101"]
TORCHVISION_MODELS = ["mobilenet_v2"]
EFFICIENTNET_MODELS = ['efficientnet-b'+str(i) for i in range(9)]


def get_model_with_linear(name: str, pretrained: str = "imagenet", last_linear_size: int = 10,
                          checkpoint: Optional[str] = None) -> Tuple[nn.Module, Dict]:
    """
    :param name:
    :param pretrained:
    :param last_linear_size:
    :param checkpoint:
    :return:
    """
    # imagenet settings, redefine if necessary
    settings = {'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]}

    if checkpoint is not None:
        print(f"Variable `checkpoint` not empty, then set `pretrained` to None")
        pretrained = None

    if name in PRETRAINED_MODELS:
        model = getattr(pretrained_models, name)(pretrained=pretrained)
        if pretrained is not None:
            settings = pretrained_settings[name][pretrained]
        del model.fc  # remove this parameter for tracing
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, last_linear_size)
    elif name in TORCHVISION_MODELS:
        model = getattr(torch_models, name)(pretrained=(pretrained == "imagenet"))
        if name == "mobilenet_v2":
            model.classifier = nn.Linear(model.last_channel, last_linear_size)
    elif name in EFFICIENTNET_MODELS:
        if pretrained:
            model = EfficientNet.from_pretrained(name)
        else:
            model = EfficientNet.from_name(name)

        out_channels = efficientnet_utils.round_filters(1280, model._global_params)
        model._fc = nn.Linear(out_channels, last_linear_size)
    else:
        raise Exception(f"Model {name} not found.")

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    return model, settings


def get_keypoints_model(name: str, pretrained: str = "imagenet", num_points: int = 10,
                        checkpoint: Optional[str] = None)\
        -> Tuple[nn.Module, Dict]:
    """
    Return model and settings for keypoints task.
    """
    return get_model_with_linear(name=name, pretrained=pretrained,
                                 last_linear_size=num_points, checkpoint=checkpoint)


def get_classification_model(name: str, pretrained: str = "imagenet", num_classes: int = 10,
                             checkpoint: Optional[str] = None)\
        -> Tuple[nn.Module, Dict]:
    """
    Return model and settings for classification.
    @TODO add last activation layer
    """
    return get_model_with_linear(name=name, pretrained=pretrained,
                                 last_linear_size=num_classes, checkpoint=checkpoint)


def get_embedding_model(name: str, pretrained: str = "imagenet", embedding_size: int = 10,
                        checkpoint: Optional[str] = None)\
        -> Tuple[nn.Module, Dict]:
    """
    Return model and settings for embedding.
    """
    return get_model_with_linear(name=name, pretrained=pretrained,
                                 last_linear_size=embedding_size, checkpoint=checkpoint)


def get_segmentation_model(name: str, encoder_name: str, pretrained: str = "imagenet", checkpoint: Optional[str] = None,
                           classes: int = 1, activation=None)\
        -> Tuple[nn.Module, Dict]:
    """
    Get segmentation model from https://github.com/qubvel/segmentation_models.pytorch
    :return:
    """
    settings = {'input_size': [3, 224, 224]}

    model = getattr(smp, name)(encoder_name=encoder_name, classes=classes,
                               encoder_weights=pretrained, activation=activation)
    settings.update(smp.encoders.get_preprocessing_params(encoder_name, pretrained))

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    return model, settings
