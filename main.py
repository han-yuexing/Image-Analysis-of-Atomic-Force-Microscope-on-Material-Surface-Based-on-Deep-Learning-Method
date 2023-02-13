#main.py

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

import torchfcn

from Data import Tai
from torch.utils import data
from train import Trainer
from net import VBN
import random

here = osp.dirname(osp.abspath(__file__))
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
   
    parser.add_argument(
        '--max-iteration', type=int, default=10000000, help='max iteration'
    )

    parser.add_argument(
        '--lr', type=float, default=1.0e-3, help='learning rate',
    )
    parser.add_argument(
        '--classes', type=int, default=2, help='classes',
    )
    
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum',
    )

    args = parser.parse_args()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    validate_indexes = random.sample(range(0, 250150), 20000)
    TrainSet = Tai(X_files = "./Train/Data", Y_files = "./Train/Label.npy", validate_indexes = validate_indexes)
    ValidSet = Tai(X_files = "./Train/Data", Y_files = "./Train/Label.npy", validate_indexes = validate_indexes, validate = True)
    train_loader = data.DataLoader(TrainSet, batch_size=4096, shuffle=True)
    valid_loader = data.DataLoader(ValidSet, batch_size=4096, shuffle=True)      

    # 2. model
    start_epoch = 0
    start_iteration = 0
    
    model = VBN(inchannel=1)
    if cuda:
        model = model.cuda()
    # print("yes model")
    # 3. optimizer

    optim = torch.optim.Adam(
        params = model.parameters(),
        lr = args.lr
        )
    # optim = torch.optim.Adam(
    #     [
    #         {'params': get_parameters(model, bias=False)},
    #         {'params': get_parameters(model, bias=True)},
    #     ],
    #     lr=args.lr,
    #     betas=(0.9, 0.99),
    #     weight_decay=args.weight_decay)
    # print("yes optimizer")

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=valid_loader,
        max_iter=args.max_iteration,
        out=args.out, 
        interval_validate=50,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
