import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import skimage.io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage
import torch.nn as nn
from sklearn.metrics import accuracy_score

def Cross_py(input, target):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([0.3, 0.7]).cuda())
    # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(input, target)
    return loss

class Trainer(object):
    def __init__(self, cuda, model, optimizer, train_loader, val_loader, max_iter, out, interval_validate=None):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'valid/loss',
            'valid/acc',
            'elapsed_time',
        ]

        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.best_acc = 0
        self.max_iter = max_iter
    
    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0

        label_trues, label_preds = [], []
        acc = 0
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            with torch.no_grad():
                score = self.model(data)
            
            loss = Cross_py(score, target.long())
            
            loss_data = loss.data.item()
            val_loss += loss_data

        
            lbl_pred = score.max(1)[1].cpu().numpy()
            # print(score, target, lbl_pred, loss)
            # break
            # print(set(lbl_pred), set(target), '\n')
            
            acc += accuracy_score(lbl_pred, target.cpu())
        acc = acc/len(self.val_loader)
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Shanghai')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 2 + [val_loss] + [acc] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        self.writer.add_scalar('acc_of_validate', acc, self.iteration/self.interval_validate)
        is_best = acc >= self.best_acc
        if is_best:
            self.best_acc = acc
            print(self.iteration)
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.writer = SummaryWriter('runs/exp2')
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()                          
            data, target = Variable(data), Variable(target)
            
            self.optim.zero_grad()
            score = self.model(data)
            loss = Cross_py(score, target.long())
            loss_data = loss.data.item()

            self.writer.add_scalar('training loss',          
                            loss_data,
                            self.iteration)

            loss.backward()
            self.optim.step()

            lbl_pred = score.max(1)[1].cpu().numpy() 
            acc = accuracy_score(lbl_pred, target.cpu())
            
            self.writer.add_scalar('acc_of_train',          
                            acc,
                            self.iteration)
                   
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Shanghai')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    [acc] + [''] * 2 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            if self.iteration >= self.max_iter:
                break

    def train(self):
        
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break