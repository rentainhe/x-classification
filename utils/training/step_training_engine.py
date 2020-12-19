import os, time, logging
import torch
import torch.nn as nn
from datasets.dataset_loader import get_train_loader
from datasets.dataset_loader import get_test_loader
from models.get_network import get_network
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.util import WarmUpLR
from criterion import LabelSmoothingCrossEntropy
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, WarmupMultiStepSchedule
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def valid(__C, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", __C.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    if __C.label_smoothing:
        loss_function = LabelSmoothingCrossEntropy(__C.smoothing)
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    for step, (images, labels) in enumerate(epoch_iterator):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            eval_outputs = model(images)
            eval_loss = loss_function(images, labels)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(eval_outputs, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], labels.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train_engine(__C):
    # define network
    net = get_network(__C)
    net = net.cuda()

    __C.batch_size = __C.batch_size // __C.gradient_accumulation_steps

    # define dataloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    # define optimizer and loss function
    if __C.label_smoothing:
        loss_function = LabelSmoothingCrossEntropy(__C.smoothing)
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=__C.lr, momentum=0.9, weight_decay=5e-4)

    # define optimizer scheduler
    # len(train_loader) 就是一个epoch的steps数量
    warmup_steps = __C.warmup_steps
    total_steps = __C.num_steps
    # change epoch into steps
    for i in __C.milestones:
        i*=len(train_loader)
    if __C.decay_type == 'multi_step':
        train_scheduler = WarmupMultiStepSchedule(__C,optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    elif __C.decay_type == 'cosine':
        train_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    elif __C.decay_type == 'linear':
        train_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    # define tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(__C.tensorboard_log_dir, __C.model, __C.version))

    # define model save dir
    checkpoint_path = os.path.join(__C.ckpts_dir, __C.model, __C.version)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{global_step}-{type}.pth')

    # define log save dir
    log_path = os.path.join(__C.result_log_dir, __C.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, __C.version + '.txt')

    # write the hyper parameters to log
    logfile = open(log_path, 'a+')
    logfile.write(str(__C))
    logfile.close()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", __C.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", __C.batch_size)
    logger.info("  Gradient Accumulation steps = %d", __C.gradient_accumulation_steps)

    net.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        net.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            loss = loss_function(images, labels)
            if __C.gradient_accumulation_steps > 1:
                loss = loss / __C.gradient_accumulation_steps
            else:
                loss.backward()

            if (step + 1) % __C.gradient_accumulation_steps == 0:
                losses.update(loss.item() * __C.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(net.parameters(), __C.max_grad_norm)
                train_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, total_steps, losses.val)
                )

                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=train_scheduler.get_lr()[0], global_step=global_step)

                if global_step % __C.eval_every == 0:
                    accuracy = valid(__C, model=net, writer=writer, test_loader=test_loader, global_step=global_step)
                    if best_acc < accuracy:
                        torch.save(net.state_dict(), checkpoint_path.format(net=__C.model, global_step=global_step, type='best'))
                        best_acc = accuracy
                    net.train()

                if global_step % total_steps == 0:
                    break
        losses.reset()
        if global_step % total_steps == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")





