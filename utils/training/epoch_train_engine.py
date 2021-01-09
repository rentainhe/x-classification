import os, time
import torch
import torch.nn as nn
from datasets.dataset_loader import get_train_loader
from datasets.dataset_loader import get_test_loader
from models.get_network import get_network
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.util import WarmUpLR
from criterion import LabelSmoothingCrossEntropy
from utils.util import split_weights

def train_engine(__C):
    # define network
    net = get_network(__C)
    net = net.cuda()

    if __C.n_gpu > 1 :
        net = nn.DataParallel(net, device_ids=__C.devices)

    # define dataloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    # define optimizer and loss function
    if __C.label_smoothing:
        loss_function = LabelSmoothingCrossEntropy(__C.smoothing)
    else:
        loss_function = nn.CrossEntropyLoss()

    # define optimizer and training parameters
    if __C.no_bias_decay:
        params = split_weights(net)
    else:
        params = net.parameters()
    optimizer = optim.SGD(params, lr=__C.lr, momentum=0.9, weight_decay=5e-4)

    # define optimizer scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=__C.milestones, gamma=__C.lr_decay_rate)
    iter_per_epoch = len(train_loader)
    warmup_schedule = WarmUpLR(optimizer, iter_per_epoch * __C.warmup_epoch)

    # define tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(__C.tensorboard_log_dir,__C.model,__C.version))

    # define model save dir
    checkpoint_path = os.path.join(__C.ckpts_dir, __C.model, __C.version)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # define log save dir
    log_path = os.path.join(__C.result_log_dir, __C.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path,__C.version+'.txt')

    # write the hyper-parameters to log
    logfile = open(log_path, 'a+')
    logfile.write(str(__C))
    logfile.close()

    # if using pytorch-mixed_up-training create a scalar
    if __C.mixed_training:
        scalar = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    loss_sum = 0
    for epoch in range(1, __C.epoch):
        if epoch > __C.warmup_epoch:
            train_scheduler.step(epoch)

        start = time.time()
        net.train()
        for step, (images, labels) in enumerate(train_loader):
            if epoch <= __C.warmup_epoch:
                warmup_schedule.step()
            images = images.cuda()
            labels = labels.cuda()
            # using gradient accumulation step

            optimizer.zero_grad()
            loss_tmp = 0
            for accu_step in range(__C.gradient_accumulation_steps):
                loss_tmp = 0
                sub_images = images[accu_step * __C.sub_batch_size:
                                    (accu_step + 1) * __C.sub_batch_size]
                sub_labels = labels[accu_step * __C.sub_batch_size:
                                    (accu_step + 1) * __C.sub_batch_size]

                if __C.mixed_training:
                    with torch.cuda.amp.autocast():
                        outputs = net(sub_images)
                        loss = loss_function(outputs, sub_labels)
                        loss = loss / __C.gradient_accumulation_steps
                    scalar.scale(loss).backward()
                else:
                    outputs = net(sub_images)
                    loss = loss_function(outputs, sub_labels)
                    loss = loss / __C.gradient_accumulation_steps
                    loss.backward()
                # loss_tmp += loss.cpu().data.numpy() * __C.gradient_accumulation_steps
                # loss_sum += loss.cpu().data.numpy() * __C.gradient_accumulation_steps
                loss_tmp += loss.cpu().data.numpy()
                loss_sum += loss.cpu().data.numpy()

            if __C.mixed_training:
                scalar.step(optimizer)
                scalar.update()
            else:
                optimizer.step()

            n_iter = (epoch-1) * len(train_loader) + step + 1
            print(
                '[{Version}] [{Model}] Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss_tmp,
                    optimizer.param_groups[0]['lr'],
                    Version=__C.version,
                    Model=__C.model,
                    epoch=epoch,
                    trained_samples=step * __C.batch_size + len(images),
                    total_samples=len(train_loader.dataset)
                ))
            # update training loss for each iteration

            writer.add_scalar('[Epoch] Train/loss', loss_tmp, n_iter)
            if epoch <= __C.warmup_epoch:
                writer.add_scalar('[Epoch] Train/lr', warmup_schedule.get_lr()[0], epoch)
            else:
                writer.add_scalar('[Epoch] Train/lr', train_scheduler.get_lr()[0], epoch)

        # update the result logfile
        logfile = open(log_path, 'a+')
        logfile.write(
            'Epoch: ' + str(epoch) +
            ', Train Average Loss: {:.4f}'.format(loss_sum/len(train_loader.dataset)) +
            ', Lr: {:.6f}'.format(optimizer.param_groups[0]['lr']) +
            ', '
        )
        logfile.close()
        finish = time.time()
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

        if __C.eval_every_epoch:
            start = time.time()
            net.eval()
            test_loss = 0.0
            correct = 0.0
            for (images, labels) in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                eval_outputs = net(images)
                eval_loss = loss_function(eval_outputs, labels)
                test_loss += eval_loss.item()
                _, preds = eval_outputs.max(1)
                correct += preds.eq(labels).sum()
            finish = time.time()

            test_average_loss = test_loss / len(test_loader.dataset)  # 测试平均 loss
            acc = correct.float() / len(test_loader.dataset)  # 测试准确率

            # save model after every "save_epoch" epoches and model with the best acc
            if epoch > __C.milestones[1] and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(net=__C.model, epoch=epoch, type='best'))
                best_acc = acc
                continue
            if not epoch % __C.save_epoch:
                torch.save(net.state_dict(), checkpoint_path.format(net=__C.model, epoch=epoch, type='regular'))

            # print the testing information
            print('Evaluating Network.....')
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                test_average_loss,
                acc,
                finish - start
            ))
            print()

            # update the result logfile
            logfile = open(log_path, 'a+')
            logfile.write(
                'Test Average loss: {:.4f}'.format(test_average_loss) +
                ', Accuracy: {:.4f}'.format(acc) +
                '\n'
            )
            logfile.close()

            # update the tensorboard log file
            writer.add_scalar('[Epoch] Test/Average loss', test_average_loss, epoch)
            writer.add_scalar('[Epoch] Test/Accuracy', acc, epoch)


