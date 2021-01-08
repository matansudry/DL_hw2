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
import torch.nn.functional as F


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
    """scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)"""
    def ExponentialLR_decay(step):
        lr = (0.5**(step/50000))*0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    loss_func = nn.LogSoftmax(dim=1).to("cuda")
    steps = 1
    for epoch in tqdm(range(train_params.num_epochs)):
        step =+ 1
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        i = 0
        for i, x in enumerate(train_loader):
            img = x[0]
            ans = x[1]
            ques = x[2]
            if torch.cuda.is_available():
                img = img.cuda()
                ans = ans.cuda()
                ques = ques.cuda()
                
            optimizer.zero_grad()
            y_hat = model((img, ques))
            img_size = img.size(0)
            nll = -F.log_softmax(y_hat)
            batch_score = train_utils.batch_accuracy(y_hat, ans.data).sum()
            ans = answer_norm(ans)
            loss = (nll * ans).sum(dim=1).mean()


            # Optimization step
            loss.backward()
            optimizer.step()
            ExponentialLR_decay(steps)

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            metrics['train_score'] += batch_score.item()
            metrics['train_loss'] += loss.item() * img_size

#             # Report model to tensorboard
#             if epoch == 0 and i == 0:
# #                 list_input = [img, ques, q_len]
#                 logger.report_graph(model, (img, ques, q_len))
            i += 1

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)
        metrics['train_score'] *= 100

        norm = metrics['total_norm'] / metrics['count_norm']

        model.eval()
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
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

    loss_func = nn.LogSoftmax(dim=1).to("cuda")

    for i, x in enumerate(dataloader):
        img = x[0]
        ans = x[1]
        ques = x[2]
        if torch.cuda.is_available():
            img = img.cuda()
            ans = ans.cuda()
            ques = ques.cuda()

        y_hat = model((img, ques))
        img = None
        ques = None
        nll = -loss_func(y_hat)
        score += train_utils.batch_accuracy(y_hat, ans.data).sum()
        ans = answer_norm(ans)
        loss += (nll * ans).sum(dim=1).mean()


    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100

    print("val loss = ", loss)

    return score, loss

def answer_norm(ans):
    zeros = torch.zeros(ans.shape[1])
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
