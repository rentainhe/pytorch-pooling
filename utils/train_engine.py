from Model.vgg import vgg16_bn
from dataset_loader import get_train_loader
from dataset_loader import get_test_loader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.util import WarmUpLR
from torch.utils.tensorboard import SummaryWriter
import time
def train_engine(__C):
    net = vgg16_bn()
    net = net.cuda()

    # define dataloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    # define optimizer and loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=__C.lr, momentum=0.9, weight_decay=5e-4)

    # define optimizer scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)
    iter_per_epoch = len(train_loader)
    warmup_schedule = WarmUpLR(optimizer, iter_per_epoch * __C.warmup_epoch)

    # define tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(__C.tensorboard_log_dir,__C.model,__C.version))

    # define model save dir
    checkpoint_path = os.path.join(__C.ckpts_dir, __C.model, __C.version)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # define the log save dir
    log_path = os.path.join(__C.result_log_dir, __C.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, __C.version + '.txt')

    # write the hyper parameters to log
    logfile = open(log_path, 'a+')
    logfile.write(str(__C))
    logfile.close()

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
                outputs = net(sub_images)
                loss = loss_function(outputs, sub_labels)
                loss.backward()
                # loss_tmp += loss.cpu().data.numpy() * __C.gradient_accumulation_steps
                # loss_sum += loss.cpu().data.numpy() * __C.gradient_accumulation_steps
                loss_tmp += loss.cpu().data.numpy()
                loss_sum += loss.cpu().data.numpy()

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

