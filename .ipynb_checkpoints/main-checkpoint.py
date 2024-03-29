"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from train import train
from dataset import MyDataset, CocoImages
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils, data_loader
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import os


torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    print("current dict = ", os.getcwd())
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load dataset
    train_dataset = MyDataset(image_path='data/train2014',
                              questions_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                              answers_path='data/v2_mscoco_train2014_annotations.json',
                              train=True,
                              answerable_only = False
                             )
    val_dataset = MyDataset(image_path='data/val2014',
                          questions_path='data/v2_OpenEnded_mscoco_val2014_questions.json',
                          answers_path='data/v2_mscoco_val2014_annotations.json',
                          train=False,
                          answerable_only = False
                         )

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                             num_workers=cfg['main']['num_workers'])
    raise

    # Init model
    model = MyModel(num_hid=cfg['train']['num_hid'], dropout=cfg['train']['dropout'])

    # TODO: Add gpus_to_use
    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, eval_loader, train_params, logger)
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
