import torch
import hydra

from train import train
from dataset2 import MyDataset, CocoImages
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils#, data_loader
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import os

import pickle

import time
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger

def main():

    train_dataset = MyDataset(image_path='../../../datashare/train2014',
                              questions_path='../../../datashare/v2_OpenEnded_mscoco_train2014_questions.json',
                              answers_path='../../../datashare/v2_mscoco_train2014_annotations.json',
                              train=True,
                              answerable_only = False
                             )
    val_dataset = MyDataset(image_path='../../../datashare/val2014',
                              questions_path='../../../datashare/v2_OpenEnded_mscoco_val2014_questions.json',
                              answers_path='../../../datashare/v2_mscoco_val2014_annotations.json',
                              train=False,
                              answerable_only = False
                             )

if __name__ == '__main__':
    main()