import os
import logging
import argparse
from typing import Dict, List, Any
from runpy import run_path

from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def configs_checker(main_config: Dict, stages_config: List[Dict]):
    """
    Configs validation.
    """
    # Main config
    required_fields = ["model", "logdir"]
    for required_field in required_fields:
        if required_field not in main_config:
            raise Exception(f"Field {required_field} is required for main config")

    # Stages
    first_stage_config = stages_config[0]
    required_fields = ["train_dataset", "val_dataset", "criterion", "optimizer"]
    for required_field in required_fields:
        if required_field not in first_stage_config:
            raise Exception(f"Field {required_field} is required for first stage config")


def train(main_config: Dict, stages_config: List[Dict]):
    """
    Train several stages.
    :param main_config:
    :param stages_config:
    :return:
    """
    configs_checker(main_config, stages_config)

    model = main_config["model"]
    model.train()

    logdir = main_config["logdir"]
    os.mkdir(logdir)

    num_workers = 0
    batch_size = 1
    num_epochs = 1
    train_dataset = None
    val_dataset = None
    callbacks = None
    batch_sampler = None

    criterion = None
    optimizer = None
    scheduler = None

    for i, config in enumerate(stages_config):
        stage_logdir = os.path.join(logdir, f"stage_{i}")

        logging.info(f"Config number = {i}, logdir = {stage_logdir}")

        num_workers = config.get("num_workers") or num_workers
        batch_size = config.get("batch_size") or batch_size
        num_epochs = config.get("num_epochs") or num_epochs
        batch_sampler = config.get("batch_sampler") or batch_sampler

        train_dataset = config.get("train_dataset") or train_dataset
        val_dataset = config.get("val_dataset") or val_dataset
        train_transforms = config.get("train_transforms")
        val_transforms = config.get("val_transforms")
        if train_transforms is not None:
            train_dataset.set_transforms(train_transforms)
        if val_transforms is not None:
            val_dataset.set_transforms(val_transforms)
        logging.info(f"Length train dataset = {len(train_dataset)}")
        logging.info(f"Length valid dataset = {len(val_dataset)}")

        loaders = dict(train=DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=(not batch_sampler), num_workers=num_workers,
                                        batch_sampler=batch_sampler),
                       valid=DataLoader(val_dataset, batch_size=batch_size,
                                        batch_sampler=batch_sampler,
                                        shuffle=False, num_workers=num_workers))

        criterion = config.get("criterion") or criterion
        optimizer = config.get("optimizer") or optimizer
        scheduler = config.get("scheduler") or scheduler
        callbacks = config.get("callbacks") or callbacks

        runner = SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders=loaders,
            logdir=stage_logdir,
            num_epochs=num_epochs,
            verbose=True,
        )


if __name__ == '__main__':
    args = arg_parse()

    config_module = run_path(args.config_file)
    main_config = config_module["main_config"]
    stages_config = config_module["stages_config"]

    train(main_config=main_config, stages_config=stages_config)
