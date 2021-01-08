import torch
import hydra

from train import train
from dataset import MyDataset
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import os
import time
import torch.nn as nn
from tqdm import tqdm
from utils.train_utils import TrainParams
import torch.nn.functional as F


@hydra.main(config_path="config", config_name='config')
def evaluate_hw2(cfg: DictConfig):
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    print(logger)
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])
    
 
    # Load dataset
    val_dataset = MyDataset(image_path=cfg['main']['paths']['val_images'],
                              questions_path=cfg['main']['paths']['val_qeustions'],
                              answers_path=cfg['main']['paths']['val_answers'],
                              train=False,
                              answerable_only = False
                             )
                            
    dataloader = DataLoader(val_dataset, 256, shuffle=False,
                             num_workers=cfg['main']['num_workers'])

    # Init model
    image_in_size_input = ((3,224,224))
    img_encoder_channels_input = [64, 64, 128, 128, 256, 256, 512]

    model = MyModel(image_in_size=image_in_size_input,
        img_encoder_out_classes=cfg['train']['img_encoder_out_classes'],
        img_encoder_channels=img_encoder_channels_input,
        img_encoder_batchnorm=cfg['train']['img_encoder_batchnorm'],
        img_encoder_dropout=cfg['train']['img_encoder_dropout'],
        text_embedding_tokens=cfg['train']['text_embedding_tokens'],
        text_embedding_features=cfg['train']['text_embedding_features'],
        text_lstm_features=cfg['train']['text_lstm_features'],
        text_dropout=cfg['train']['text_dropout'],
        classifier_dropout=cfg['train']['classifier_dropout'],
        classifier_mid_features=cfg['train']['classifier_mid_features'],
        classifier_out_classes=val_dataset.number_of_answers_per_question
        )

    convert_file('model.pth')
    #model.load_state_dict(torch.load('model.pkl',map_location=lambda storage, loc: storage))
    temp = torch.load('model.pkl',map_location='cpu')
    model.load_state_dict(temp)
    
    print("len = ",len(dataloader.dataset))

    model.eval()
    """if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)"""
    print(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))


    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    score = 0
    loss = 0
    if torch.cuda.is_available():
        loss_func = nn.LogSoftmax(dim=1).to("cuda")
    else:
        loss_func = nn.LogSoftmax(dim=1)
    for i, x in enumerate(tqdm(dataloader)):
        img = x[0]
        ans = x[1]
        ques = x[2]
        if torch.cuda.is_available():
            img = img.cuda()
            ans = ans.cuda()
            ques = ques.cuda()
        with torch.no_grad():
            y_hat = model((img, ques))
        img = None
        ques = None
        nll = -loss_func(y_hat)
        score += train_utils.batch_accuracy(y_hat, ans.data).sum()
        print(score)
        ans = answer_norm(ans)
        loss += (nll * ans).sum(dim=1).mean()
        ans = None


    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100
    print(score.item())

    return score.item()

def answer_norm(ans):
    zeros = torch.zeros(ans.shape[1])
    if torch.cuda.is_available():
        zeros = zeros .cuda()
    max_values, predicted_index = ans.max(dim=1, keepdim=True)
    for i in range(ans.shape[0]):
        ans[i] = torch.where(ans[i]==max_values[i], ans[i], zeros)
    ans_sum = ans.sum(dim=1)
    for i in range(len(ans_sum)):
        if ans_sum[i] == 0:
            ans_sum[i]=1
        ans[i] = ans[i]/ans_sum[i]
    return ans

def convert_file(path):
    old_dict = torch.load(path ,map_location=lambda storage, loc: storage)
    reduced_dict = old_dict['model_state']
    old = ["module.fc1.weight", "module.fc1.bias", "module.fc2.weight", "module.fc2.bias",
       "module.img_encoder.feature_extractor.0.weight", "module.img_encoder.feature_extractor.1.weight",
       "module.img_encoder.feature_extractor.1.bias", "module.img_encoder.feature_extractor.1.running_mean",
       "module.img_encoder.feature_extractor.1.running_var",
       "module.img_encoder.feature_extractor.4.main_path.0.weight", "module.img_encoder.feature_extractor.4.main_path.0.bias",
       "module.img_encoder.feature_extractor.4.main_path.1.weight", "module.img_encoder.feature_extractor.4.main_path.1.bias",
       "module.img_encoder.feature_extractor.4.main_path.1.running_mean", "module.img_encoder.feature_extractor.4.main_path.1.running_var",
       "module.img_encoder.feature_extractor.5.main_path.0.weight",
       "module.img_encoder.feature_extractor.5.main_path.0.bias", "module.img_encoder.feature_extractor.5.main_path.1.weight",
       "module.img_encoder.feature_extractor.5.main_path.1.bias", "module.img_encoder.feature_extractor.5.main_path.1.running_mean",
       "module.img_encoder.feature_extractor.5.main_path.1.running_var",
       "module.img_encoder.feature_extractor.6.main_path.0.weight", "module.img_encoder.feature_extractor.6.main_path.0.bias",
       "module.img_encoder.feature_extractor.6.main_path.1.weight", "module.img_encoder.feature_extractor.6.main_path.1.bias",
       "module.img_encoder.feature_extractor.6.main_path.1.running_mean", "module.img_encoder.feature_extractor.6.main_path.1.running_var", "module.img_encoder.feature_extractor.7.main_path.0.weight",
       "module.img_encoder.feature_extractor.7.main_path.0.bias", "module.img_encoder.feature_extractor.7.main_path.1.weight", 
       "module.img_encoder.feature_extractor.7.main_path.1.bias", "module.img_encoder.feature_extractor.7.main_path.1.running_mean",
       "module.img_encoder.feature_extractor.7.main_path.1.running_var",
       "module.img_encoder.feature_extractor.8.main_path.0.weight", "module.img_encoder.feature_extractor.8.main_path.0.bias", 
       "module.img_encoder.feature_extractor.8.main_path.1.weight", "module.img_encoder.feature_extractor.8.main_path.1.bias", 
       "module.img_encoder.feature_extractor.8.main_path.1.running_mean", "module.img_encoder.feature_extractor.8.main_path.1.running_var", "module.img_encoder.feature_extractor.9.main_path.0.weight",
       "module.img_encoder.feature_extractor.9.main_path.0.bias", "module.img_encoder.feature_extractor.9.main_path.1.weight",
       "module.img_encoder.feature_extractor.9.main_path.1.bias", "module.img_encoder.feature_extractor.9.main_path.1.running_mean",
       "module.img_encoder.feature_extractor.9.main_path.1.running_var",
       "module.img_encoder.feature_extractor.10.main_path.0.weight", "module.img_encoder.feature_extractor.10.main_path.0.bias",
       "module.img_encoder.feature_extractor.10.main_path.1.weight", "module.img_encoder.feature_extractor.10.main_path.1.bias",
       "module.img_encoder.feature_extractor.10.main_path.1.running_mean", "module.img_encoder.feature_extractor.10.main_path.1.running_var", "module.img_encoder.fc1.weight", "module.img_encoder.fc1.bias",
       "module.img_encoder.fc2.weight", "module.img_encoder.fc2.bias", "module.text.embedding.weight", "module.text.lstm.weight_ih_l0",
       "module.text.lstm.weight_hh_l0", "module.text.lstm.bias_ih_l0", "module.text.lstm.bias_hh_l0", "module.text.lstm.weight_ih_l1",
       "module.text.lstm.weight_hh_l1", "module.text.lstm.bias_ih_l1", "module.text.lstm.bias_hh_l1", "module.text.fc.weight", "module.text.fc.bias"]

    new = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "img_encoder.feature_extractor.0.weight", "img_encoder.feature_extractor.1.weight",
           "img_encoder.feature_extractor.1.bias", "img_encoder.feature_extractor.1.running_mean", "img_encoder.feature_extractor.1.running_var", 
           "img_encoder.feature_extractor.4.main_path.0.weight", "img_encoder.feature_extractor.4.main_path.0.bias",
           "img_encoder.feature_extractor.4.main_path.1.weight", "img_encoder.feature_extractor.4.main_path.1.bias",
           "img_encoder.feature_extractor.4.main_path.1.running_mean", "img_encoder.feature_extractor.4.main_path.1.running_var",
           "img_encoder.feature_extractor.5.main_path.0.weight", "img_encoder.feature_extractor.5.main_path.0.bias",
           "img_encoder.feature_extractor.5.main_path.1.weight", "img_encoder.feature_extractor.5.main_path.1.bias",
           "img_encoder.feature_extractor.5.main_path.1.running_mean", "img_encoder.feature_extractor.5.main_path.1.running_var",
           "img_encoder.feature_extractor.6.main_path.0.weight", "img_encoder.feature_extractor.6.main_path.0.bias", 
           "img_encoder.feature_extractor.6.main_path.1.weight", "img_encoder.feature_extractor.6.main_path.1.bias",
           "img_encoder.feature_extractor.6.main_path.1.running_mean", "img_encoder.feature_extractor.6.main_path.1.running_var",
           "img_encoder.feature_extractor.7.main_path.0.weight", "img_encoder.feature_extractor.7.main_path.0.bias", 
           "img_encoder.feature_extractor.7.main_path.1.weight", "img_encoder.feature_extractor.7.main_path.1.bias", 
           "img_encoder.feature_extractor.7.main_path.1.running_mean", "img_encoder.feature_extractor.7.main_path.1.running_var", 
           "img_encoder.feature_extractor.8.main_path.0.weight", "img_encoder.feature_extractor.8.main_path.0.bias", 
           "img_encoder.feature_extractor.8.main_path.1.weight", "img_encoder.feature_extractor.8.main_path.1.bias", 
           "img_encoder.feature_extractor.8.main_path.1.running_mean", "img_encoder.feature_extractor.8.main_path.1.running_var",
           "img_encoder.feature_extractor.9.main_path.0.weight", "img_encoder.feature_extractor.9.main_path.0.bias", 
           "img_encoder.feature_extractor.9.main_path.1.weight", "img_encoder.feature_extractor.9.main_path.1.bias",
           "img_encoder.feature_extractor.9.main_path.1.running_mean", "img_encoder.feature_extractor.9.main_path.1.running_var",
           "img_encoder.feature_extractor.10.main_path.0.weight", "img_encoder.feature_extractor.10.main_path.0.bias", 
           "img_encoder.feature_extractor.10.main_path.1.weight", "img_encoder.feature_extractor.10.main_path.1.bias", 
           "img_encoder.feature_extractor.10.main_path.1.running_mean", "img_encoder.feature_extractor.10.main_path.1.running_var", 
           "img_encoder.fc1.weight", "img_encoder.fc1.bias", "img_encoder.fc2.weight", "img_encoder.fc2.bias", "text.embedding.weight", 
           "text.lstm.weight_ih_l0", "text.lstm.weight_hh_l0", "text.lstm.bias_ih_l0", "text.lstm.bias_hh_l0", "text.lstm.weight_ih_l1", 
           "text.lstm.weight_hh_l1", "text.lstm.bias_ih_l1", "text.lstm.bias_hh_l1", "text.fc.weight", "text.fc.bias"]
    new_dict = {}
    for i in range(len(old)):
        new_dict[new[i]] = reduced_dict[old[i]]
    torch.save(new_dict, 'model.pkl')

if __name__ == '__main__':
    evaluate_hw2()

