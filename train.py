"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn

from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)
    loss_func = nn.LogSoftmax()
    for epoch in tqdm(range(train_params.num_epochs)):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        
        i = 0
        for img, ans, ques, _, q_len in train_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                ans = ans.cuda()
                ques = ques.cuda()
                q_len = q_len.cuda()
                

            y_hat = model((img, ques, q_len))
            img_size = img.size(0)
            img = None
            ques = None
            q_len = None
            nll = -loss_func(y_hat)
            loss = (nll * ans).sum(dim=1).mean()


            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            # NOTE! This function compute scores correctly only for one hot encoding representation of the logits
            batch_score = train_utils.batch_accuracy(y_hat, ans.data).sum()
            metrics['train_score'] += batch_score.item()

            metrics['train_loss'] += loss.item() * img_size

#             # Report model to tensorboard
#             if epoch == 0 and i == 0:
# #                 list_input = [img, ques, q_len]
#                 logger.report_graph(model, (img, ques, q_len))
            i += 1
        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)
        metrics['train_score'] *= 100

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['train_loss'],
                   'Loss/Train': metrics['eval_score'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)

        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0

    for img, ans, ques, _, q_len in dataloader:
        if torch.cuda.is_available():
            img = img.cuda()
            ans = ans.cuda()
            ques = ques.cuda()
            q_len = q_len.cuda()

        y_hat = model((img, ques, q_len))
        img = None
        ques = None
        q_len = None
        loss += nn.functional.binary_cross_entropy_with_logits(y_hat, ans)
        score += train_utils.compute_score_with_logits(y_hat, ans).sum().item()

    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100

    return score, loss
