import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf import settings
from util import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

def train_engine(config):
    # data processing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=config.batch_size,
        shuffle=True
    )

    net = get_network(config)
    net = net.cuda()
    net = net.train()

    if config.gpu_nums > 1:
        net = nn.DataParallel(net, device_ids=config.devices)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.optim_params['momentum'], weight_decay=config.optim_params['weight_decay'])
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.warmup_epoch)

    if not os.path.exists(config.tensorboard_log_dir):
        os.mkdir(config.tensorboard_log_dir)

    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, config.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    if config.resume:
        print("========== Resume training ==========")

        if config.ckpt_path is not None:
            path = config.ckpt_path
        else:
            path = config.ckpts_path + '/' + \
                   str(config.net) + '/ckpt_' + config.ckpt_version + \
                   'epoch' + str(config.ckpt_epoch) + '.pkl'

        # Load the network parameters
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

    # 将超参写入 log 文件
    logfile = open(
        config.result_logfile_path +
        '/' + str(config.net) + '/log_run_' + config.version + '.txt',
        'a+'
    )
    logfile.write(str(config))
    logfile.close()

    for epoch in range(1, config.epoch):
        if epoch > config.warmup_epoch:
            train_scheduler.step(epoch)

        start = time.time()
        net.train()
        for step, (images, labels) in enumerate(cifar100_training_loader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            loss_tmp = 0
            for accu_step in range(config.gradient_accumulation_steps):
                loss_tmp = 0
                sub_images = images[accu_step * config.sub_batch_size:
                                    (accu_step+1) * config.sub_batch_size]
                sub_labels = labels[accu_step * config.sub_batch_size:
                                    (accu_step+1) * config.sub_batch_size]
                outputs = net(sub_images)
                loss_item = [outputs, sub_labels]
                loss = loss_function(loss_item[0], loss_item[1])
                loss /= config.gradient_accumulation_steps
                loss.backward()
                loss_tmp += loss.cpu().data.numpy() * config.gradient_accumulation_steps

            optimizer.step()
            n_iter = (epoch-1) * len(cifar100_training_loader) + step + 1
            print('[{Version}] [{Model}] Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss_tmp,
                optimizer.param_groups[0]['lr'],
                Version=config.version,
                Model = config.model,
                epoch=epoch,
                trained_samples=step * config.batch_size + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

            # update training loss for each iteration
            writer.add_scalar('Train/loss', loss_tmp, n_iter)

@torch.no_grad()
def eval_training(net, config,epoch=0, tb=True):

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=config.batch_size,
        shuffle=False
    )

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:


        images = images.cuda()
        labels = labels.cuda()
        loss_function = nn.CrossEntropyLoss()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('GPU INFO.....')
    # print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()
