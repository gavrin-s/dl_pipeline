import os
import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from catalyst.dl.callbacks import AccuracyCallback

from src.config import DATA_PATH, LOG_PATH
from src.utils import parse_cvat_xml
from src.datasets import KeyPointsDataset, CustomConcatDataset, ClassificationDataset
from src.models import get_classification_model

# --- Standard settings  --- #
model_name = "resnet50"
caching = True
num_workers = 12
batch_size = 64
logdir = os.path.join(LOG_PATH, "classification-resnet50")

# --- Define model and is't settings  --- #
model, settings = get_classification_model(model_name, num_classes=52)
target_size = settings["input_size"][1:3]

# --- Preprocessing and Postprocessing  --- #
preprocessing = albu.Compose([
    albu.PadIfNeeded(960, 960, border_mode=0),
    albu.Resize(*target_size),
], keypoint_params=albu.KeypointParams(format='xy'))

postprocessing = albu.Compose([
    albu.Normalize(mean=settings["mean"], std=settings["std"]),
    ToTensorV2(),
])

# --- Define datasets  --- #
labels = []
keypoint_datasets = []

datapath = os.path.join(DATA_PATH, "photo")
dataset_names = os.listdir(datapath)

for label, dataset_name in enumerate(dataset_names):
    annotation = parse_cvat_xml(os.path.join(datapath, dataset_name, "hands.xml"))
    annotation_dict = annotation.set_index("name")["points"].to_dict()

    keypoint_dataset = KeyPointsDataset(annotation_dict,
                                        os.path.join(datapath, dataset_name),
                                        caching=False,
                                        preprocessing=preprocessing,
                                        )

    labels.extend([label] * len(keypoint_dataset))
    keypoint_datasets.append(keypoint_dataset)

labels = np.array(labels)
keypoint_dataset = CustomConcatDataset(keypoint_datasets)
train_dataset = ClassificationDataset(keypoint_dataset, labels,
                                      postprocessing=postprocessing)

# --- Define optimizers  --- #
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)

main_config = dict(
    model=model,
    logdir=logdir
)

stages_config = [
    dict(num_workers=num_workers,
         batch_size=batch_size,
         num_epochs=150,
         train_dataset=train_dataset,
         val_dataset=train_dataset,
         criterion=torch.nn.NLLLoss(),
         optimizer=optimizer,
         scheduler=scheduler,
         callbacks=[AccuracyCallback(prefix="accuracy")],
         ),
]
