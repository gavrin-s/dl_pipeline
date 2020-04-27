import os
import torch

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from catalyst.dl.callbacks import MetricCallback
from src.config import DATA_PATH, LOG_PATH
from src.models import get_keypoints_model
from src.datasets import CustomConcatDataset, KeyPointsDataset
from src.transforms import KeyPointNormalize, RandomMAxSquaredCrop
from src.utils import parse_cvat_xml

model_name = "SimpleCNN2"
caching = True
num_workers = 12
batch_size = 64
logdir = os.path.join(LOG_PATH, "test-pipeline")

model, settings = get_keypoints_model(model_name)
target_size = settings["input_size"][1:3]

# --- Preprocessing and Postprocessing  --- #
preprocessing = albu.Compose([RandomMAxSquaredCrop(),
                              albu.Resize(*target_size)],
                             keypoint_params=albu.KeypointParams(format='xy'))

postprocessing = albu.Compose([
    KeyPointNormalize(*target_size),
    albu.Normalize(mean=settings["mean"], std=settings["std"]),
    ToTensorV2(),
], keypoint_params=albu.KeypointParams(format='xy'))

# --- Define datasets  --- #
datasets = []
for folder in os.listdir(DATA_PATH)[1:] * 5:
    try:
        annotation = parse_cvat_xml(os.path.join(DATA_PATH, folder, "hands.xml"), type_only="points")
    except FileNotFoundError:
        print(f"File hands.xml not found in {os.path.join(DATA_PATH, folder)}")
        continue
    annotation_dict = annotation.set_index("name")["points"].to_dict()
    datasets.append(KeyPointsDataset(annotation_dict, os.path.join(DATA_PATH, folder),
                                     preprocessing=preprocessing, postprocessing=postprocessing,
                                     caching=caching))
    print(f"Dataset {folder} loaded!")

train_dataset = CustomConcatDataset(datasets)

folder = os.listdir(DATA_PATH)[0]
annotation = parse_cvat_xml(os.path.join(DATA_PATH, folder, "hands.xml"), type_only="points")
annotation_dict = annotation.set_index("name")["points"].to_dict()
val_dataset = KeyPointsDataset(annotation_dict, os.path.join(DATA_PATH, folder),
                               preprocessing=preprocessing, postprocessing=postprocessing,
                               caching=caching)

train_transforms_simple = albu.Compose([albu.RandomRotate90(p=1),
                                        albu.Transpose(),
                                        albu.Flip(),
                                        albu.ShiftScaleRotate(shift_limit=0.3,
                                                              scale_limit=0.3,
                                                              border_mode=0)
                                        ], keypoint_params=albu.KeypointParams(format='xy')
                                       )

first_stage_optimizer = torch.optim.Adam(model.parameters())


main_config = dict(
    model=model,
    logdir=logdir
)

# define configs, one dict for one stage
stages_config = [
    dict(num_workers=num_workers,
         batch_size=batch_size,
         num_epochs=10,
         train_dataset=train_dataset,
         val_dataset=val_dataset,
         train_transforms=train_transforms_simple,
         criterion=torch.nn.MSELoss(),
         optimizer=first_stage_optimizer,
         callbacks=[MetricCallback(prefix="l1_metric", metric_fn=torch.nn.L1Loss())],
         ),
]
