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
    print(logger)
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load dataset
    train_dataset = MyDataset(image_path=cfg['main']['paths']['train_images'],
                              questions_path=cfg['main']['paths']['train_qeustions'],
                              answers_path=cfg['main']['paths']['train_answers'],
                              train=True,
                              answerable_only = False
                             )
    val_dataset = MyDataset(image_path=cfg['main']['paths']['val_images'],
                              questions_path=cfg['main']['paths']['val_qeustions'],
                              answers_path=cfg['main']['paths']['val_answers'],
                              train=False,
                              answerable_only = False
                             )
    
    
    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=False,
                             num_workers=cfg['main']['num_workers'])

    # Init model
    image_in_size_input = ((3,224,224))
    img_encoder_channels_input = [32, 128, 512, 1024]
    model = MyModel(image_in_size=image_in_size_input,
        img_encoder_out_classes=cfg['train']['img_encoder_out_classes'],
        img_encoder_channels=img_encoder_channels_input,
        img_encoder_batchnorm=cfg['train']['img_encoder_batchnorm'],
        img_encoder_dropout=cfg['train']['img_encoder_dropout'],
        text_embedding_tokens=cfg['train']['text_embedding_tokens'],
        text_embedding_features=cfg['train']['text_embedding_features'],
        text_lstm_features=cfg['train']['text_lstm_features'],
        text_dropout=cfg['train']['text_dropout'],
        attention_mid_features=cfg['train']['attention_mid_features'],
        attention_glimpses=cfg['train']['attention_glimpses'],
        attention_dropout=cfg['train']['attention_dropout'],
        classifier_dropout=cfg['train']['classifier_dropout'],
        classifier_mid_features=cfg['train']['classifier_mid_features'],
        classifier_out_classes=cfg['train']['classifier_out_classes']
        )

        

    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
