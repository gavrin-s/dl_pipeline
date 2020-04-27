import os
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from catalyst.dl.callbacks import MetricCallback, CriterionCallback, MetricAggregationCallback
import segmentation_models_pytorch as smp

from src.models import get_segmentation_model
from src.transforms import PadToMaxSize, TransposeMask
from src.datasets import SegmentationDataset, CustomConcatDataset
from src.config import DATA_PATH, LOG_PATH
from src.utils import parse_cvat_xml

model_name = "Unet"
encoder_name = "resnet50"
caching = True
num_workers = 12
batch_size = 40
logdir = os.path.join(LOG_PATH, "segmentation-unet-resnet50_5")

model, settings = get_segmentation_model(model_name, encoder_name, activation="sigmoid")
target_size = settings["input_size"][1:3]


# --- Preprocessing and Postprocessing  --- #

preprocessing = albu.Compose([
    PadToMaxSize(border_mode=0),
    albu.Resize(*target_size)
])

postprocessing = albu.Compose([
    albu.Normalize(mean=settings["mean"], std=settings["std"]),
    TransposeMask(),
    ToTensorV2(),
])

# --- Define datasets  --- #
dataset_path1 = DATA_PATH
folders1 = ["photo1", "photo2"]

dataset_path2 = os.path.join(DATA_PATH, "photo")
folders2 = os.listdir(dataset_path2)
datasets = []
for folder in folders1:
    folder = os.path.join(dataset_path1, folder)
    annotation = parse_cvat_xml(os.path.join(folder, "hands.xml"), type_only="polygon")
    annotation_dict = annotation.set_index("name")["points"].to_dict()

    dataset = SegmentationDataset(annotation_dict, folder, caching=True,
                                  preprocessing=preprocessing, postprocessing=postprocessing)
    datasets.append(dataset)


for folder in folders2:
    folder = os.path.join(dataset_path2, folder)
    if not os.path.exists(os.path.join(folder, "hands.xml")):
        continue

    annotation = parse_cvat_xml(os.path.join(folder, "hands.xml"), type_only="polygon")
    annotation_dict = annotation.set_index("name")["points"].to_dict()
    if annotation_dict:
        dataset = SegmentationDataset(annotation_dict, folder, caching=True,
                                      preprocessing=preprocessing, postprocessing=postprocessing)
        datasets.append(dataset)

train_dataset = CustomConcatDataset(datasets)
train_dataset = CustomConcatDataset([train_dataset] * 3)


# --- Define transforms  --- #
simple_transforms = albu.Compose([
    albu.RandomRotate90(p=1),
    albu.Transpose(),
    albu.Flip(),
    albu.ShiftScaleRotate(shift_limit=0.3,
                          scale_limit=0.3,
                          border_mode=0)

])

hard_transforms = albu.Compose([
    simple_transforms,
    albu.CoarseDropout(),
    albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    albu.GridDistortion(p=0.3),
    albu.HueSaturationValue(p=0.3)
])


# --- Define optimizers  --- #

criterion = {
    "dice": smp.utils.losses.DiceLoss(),
    "iou": smp.utils.losses.JaccardLoss(),
    "bce": smp.utils.losses.BCELoss()
}
optimizer = torch.optim.Adam(model.parameters())

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25)
callbacks = [
    CriterionCallback(prefix="loss_dice", criterion_key="dice"),
    CriterionCallback(prefix="loss_iou", criterion_key="iou"),
    CriterionCallback(prefix="loss_bce", criterion_key="bce"),
    MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum",
        metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
    ),

    MetricCallback("IoU_Metric", smp.utils.metrics.IoU())
]

# --- Define configs  --- #
main_config = dict(
    model=model,
    logdir=logdir
)

stages_config = [
    dict(num_workers=num_workers,
         batch_size=batch_size,
         num_epochs=100,
         train_dataset=train_dataset,
         val_dataset=train_dataset,
         train_transforms=hard_transforms,
         val_transforms=simple_transforms,
         criterion=criterion,
         optimizer=optimizer,
         callbacks=callbacks,
         scheduler=scheduler
         ),
    dict(num_epochs=50,
         train_transforms=simple_transforms,
         ),
]
