import sys
import os
import time
import numpy as np
import random
import torch
import utils
from hhutil.io import time_now

import torch.nn.functional as F
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchsummary import summary

from models.resnet import resnet18
from models.efficientnet import efficientnet_b0
from models.vit import deit_ti, swin_ti
from models.dls_darts import dls_darts_trans
from lr_scheduler import CosineLR
from loss import FocalLoss
from utils import accuracy, sensitivity, specificity, f1, auc
from transforms.operations import HFlip, VFlip, PadCrop
from transforms.EEA import ExploreExploitAugment
from transforms.AA import AutoAugment
from transforms import Normalize

# Params to replace
task_id = 1
save_path = None

# Save settings
save_ckpt_freq = None
continue_training = True

# Device settings
gpu_device_id = 0
torch.cuda.set_device(gpu_device_id)

# Basic training settings
seed = 0
grad_clip = None
use_fp16 = True
# label_smoothing = 0.0
train_optim = 'Adam'    # ['SGD', 'Adam']

loss_type = 'Focal'
# Focal loss settings
focal_alpha = [1.0, 0.33]
focal_gamma = 2.0

batch_size = 128 * 2
eval_batch_size = 64
base_lr = 1e-3
min_lr = 0
epochs = 50
warmup_epochs = 5
wd = 1e-4

# Training settings for different models
model_name = 'resnet18'
if model_name == 'resnet18':
    dropout = 0.5
elif model_name == 'efficientnet_b0':
    dropout = 0.5
    pretrained = False
elif model_name == 'deit_ti':
    dropout = 0.5
    droppath = 0.2
    pretrained = False
elif model_name == 'dls_darts':
    droppath = 0.2
else:
    raise ValueError('Model not implemented!')

# Dataset settings
NUM_CLASS = 2
RES = 128
init_c = 3
kfold = 5

if model_name == 'dls_darts':
    data_mean = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    data_std = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
else:
    data_mean = [0., 0., 0.]
    data_std = [1., 1., 1.]

# Augment settings
augment_space = 'RA-glioma'
aug_depth = 2
p_min_t = 1.0
p_max_t = 1.0

aug_type = 'EEA'
if aug_type == 'RA':
    # RA settings
    reduction = 'mean'
elif aug_type == 'EEA':
    # EEA settings
    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.1
    eea_lr = 1e-2
    gamma = 0.98
    num_bins = 11
    reduction = 'none'
elif aug_type == 'AA':
    # AA settings
    reduction = 'none'  # reuse training pipeline of EEA
else:
    raise ValueError('Augmentation %s not implemented!' % aug_type)

# CutMix settings
cutmix_beta = 0.2
cutmix_prob = 1.0


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def basic_transforms():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        PadCrop(size=RES, padding=RES // 8),
        HFlip(),
        VFlip()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    dataloaders = YOUR_DATA_LOADER
    for i_fold, (train_loader, test_loader) in enumerate(dataloaders):
        # if i_fold > 3:
        #     continue
        print('\n\n################ Start training Fold %d/%d ##################' % (i_fold + 1, kfold))
        start_time = time.time()

        if aug_type == 'EEA':
            eea = ExploreExploitAugment(depth=aug_depth, epsilon=epsilon, learning_rate=eea_lr, reward_decay=gamma,
                                        num_bins=num_bins, resolution=RES, augment_space=augment_space,
                                        p_min_t=p_min_t, p_max_t=p_max_t)
            aa = None
        elif aug_type == 'AA':
            eea = None
            aa = AutoAugment(resolution=RES)
        else:
            eea = None
            aa = None

        if loss_type == 'Focal':
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction).cuda()
        elif loss_type == 'CE':
            criterion = torch.nn.CrossEntropyLoss(reduction=reduction).cuda()
        else:
            raise NotImplementedError('Loss should be Focal or CE!')
        if model_name == 'resnet18':
            model = resnet18(num_classes=NUM_CLASS, dropout=dropout, init_c=init_c,
                             aug_depth=aug_depth,
                             resolution=RES, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                             norm_mean=data_mean, norm_std=data_std).cuda()
        elif model_name == 'efficientnet_b0':
            model = efficientnet_b0(pretrained=pretrained, num_classes=NUM_CLASS, dropout=dropout, init_c=init_c,
                             aug_depth=aug_depth,
                             resolution=RES, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                             norm_mean=data_mean, norm_std=data_std).cuda()
        elif model_name == 'deit_ti':
            model = deit_ti(pretrained=pretrained, num_classes=NUM_CLASS, dropout=dropout, droppath=droppath, init_c=init_c,
                             aug_depth=aug_depth,
                             resolution=RES, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                             norm_mean=data_mean, norm_std=data_std).cuda()
        elif model_name == 'dls_darts':
            model = dls_darts_trans(num_classes=NUM_CLASS, droppath=droppath,
                            aug_depth=aug_depth,
                            resolution=RES, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                            norm_mean=data_mean, norm_std=data_std).cuda()
        else:
            raise NotImplementedError('Model to implement!')
        # summary(model, (init_c, RES, RES), device='cuda')

        if train_optim == 'SGD':
            model_optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                              momentum=0.9, weight_decay=wd, nesterov=True)
        elif train_optim == 'Adam':
            model_optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
        else:
            raise NotImplementedError('Optimizer not implemented!')

        steps_per_epoch = len(train_loader)
        model_scheduler = CosineLR(model_optimizer, steps_per_epoch * epochs * 2, min_lr=min_lr,
                                   warmup_epoch=steps_per_epoch * warmup_epochs * 2, warmup_min_lr=0)   # *2: One iteration with 2 updates

        start_epoch = 0
        if continue_training and os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=lambda storage, loc: storage.cuda(gpu_device_id))
            model.load_state_dict(ckpt['model_state_dict'])
            model_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            model_scheduler.last_epoch += start_epoch * steps_per_epoch * 2
            print('Start training from ckpt %s.' % save_path)

        best_epoch = 0
        best_score = 0.
        best_metrics = []
        for epoch in range(start_epoch, epochs):
            print('%s Epoch %d/%d' % (time_now(), epoch + 1, epochs))
            if eea is not None:
                if eea.epsilon * epsilon_decay > epsilon_min:
                    eea.epsilon = eea.epsilon * epsilon_decay
                else:
                    eea.epsilon = epsilon_min

            # training
            train_loss, train_acc, train_spe, train_sen, train_f1, train_auc = train_epoch(
                train_loader, model, model_optimizer, model_scheduler, criterion, eea, aa)
            print('%s [Train] loss: %.4f, acc: %.4f, spe: %.4f, sen: %.4f, f1: %.4f, auc: %.4f' %
                  (time_now(), train_loss, train_acc, train_spe, train_sen, train_f1, train_auc))

            if save_ckpt_freq is not None and (epoch + 1) % save_ckpt_freq == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': model_optimizer.state_dict(),
                }, save_path)

            # inference
            val_loss, val_acc, val_spe, val_sen, val_f1, val_auc, pred_all, label_all = infer(test_loader, model, criterion, eea, aa)
            print('%s [Valid] loss: %.4f, acc: %.4f, spe: %.4f, sen: %.4f, f1: %.4f, auc: %.4f' %
                  (time_now(), val_loss, val_acc, val_spe, val_sen, val_f1, val_auc))

            if val_acc * val_spe * val_sen * val_f1 * val_auc > best_score:
                best_epoch = epoch
                best_metrics = [val_acc, val_spe, val_sen, val_f1, val_auc]
                best_score = val_acc * val_spe * val_sen * val_f1 * val_auc

        print('%s End training' % time_now())
        end_time = time.time()
        elapsed = end_time - start_time
        print('Search time: %.3f Hours' % (elapsed / 3600.))
        print('Best score (epoch %d/%d): acc: %.4f, spe: %.4f, sen: %.4f, f1: %.4f, auc: %.4f' %
              tuple([best_epoch + 1, epochs] + [metrics for metrics in best_metrics]))


def train_epoch(train_loader, model, train_optim, model_scheduler, criterion, eea, aa):
    train_loss_m = utils.AverageMeter()
    train_top1 = utils.AverageMeter()
    scaler = GradScaler(enabled=use_fp16)
    pred_all = []
    target_all = []

    for step, train_data in enumerate(train_loader):
        model.train()

        train_input, train_label = [x.cuda() for x in train_data]
        N = train_input.shape[0]

        # split two batches
        train_input_b1 = train_input[:N // 2]
        train_input_b2 = train_input[N // 2:]
        train_label_b1 = train_label[:N // 2]
        train_label_b2 = train_label[N // 2:]

        # First batch
        r = np.random.rand(1)
        if cutmix_beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(cutmix_beta, cutmix_beta)  # 随机的lam
            rand_index = torch.randperm(train_input_b1.size()[0]).to(train_input_b1.device)
            target_a = train_label_b1
            target_b = train_label_b1[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(train_input_b1.size(), lam)
            train_input_b1[:, :, bbx1:bbx2, bby1:bby2] = train_input_b1[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (train_input_b1.size()[-1] * train_input_b1.size()[-2]))

        if eea is not None or aa is not None:
            if eea is not None:
                train_input_b1 = eea(train_input_b1, use_Q_table=True)
            else:
                train_input_b1 = aa(train_input_b1)
            train_optim.zero_grad()
            with autocast(enabled=use_fp16):
                train_pred_b1 = model(train_input_b1, training=False)
                if cutmix_beta > 0 and r < cutmix_prob:
                    batch_train_loss = criterion(train_pred_b1[0], target_a.type(torch.long))
                    batch_train_loss_tmp = criterion(train_pred_b1[0], target_b.type(torch.long))
                    train_loss = torch.mean(batch_train_loss * lam + batch_train_loss_tmp * (1. - lam))
                else:
                    batch_train_loss = criterion(train_pred_b1[0], train_label_b1.type(torch.long))
                    train_loss = torch.mean(batch_train_loss)
            if eea is not None:
                eea.cal_bin_cnts(batch_train_loss)
                eea.cal_reward(batch_train_loss)
        else:
            ra_deformation = torch.ones((aug_depth, N // 2)) * 0.3
            train_optim.zero_grad()
            with autocast(enabled=use_fp16):
                train_pred_b1 = model(train_input_b1, training=True, cos=ra_deformation)
                if cutmix_beta > 0 and r < cutmix_prob:
                    train_loss = criterion(train_pred_b1[0], target_a.type(torch.long)) * lam + criterion(train_pred_b1[0], target_b.type(torch.long)) * (1. - lam)
                else:
                    train_loss = criterion(train_pred_b1[0], train_label_b1.type(torch.long))

        scaler.scale(train_loss).backward()
        if use_fp16:
            scaler.unscale_(train_optim)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(train_optim)
        scaler.update()
        model_scheduler.step()

        train_prec1 = accuracy(train_pred_b1[0].detach(), train_label_b1.detach(), topk=(1,))
        train_loss_m.update(train_loss.detach().item(), N // 2)
        train_top1.update(train_prec1.detach().item(), N // 2)
        pred_all.append(train_pred_b1[0].detach())
        target_all.append(train_label_b1)

        # Second batch
        r = np.random.rand(1)
        if cutmix_beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(cutmix_beta, cutmix_beta)
            rand_index = torch.randperm(train_input_b2.size()[0]).to(train_input_b2.device)
            target_a = train_label_b2
            target_b = train_label_b2[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(train_input_b2.size(), lam)
            train_input_b2[:, :, bbx1:bbx2, bby1:bby2] = train_input_b2[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (train_input_b2.size()[-1] * train_input_b2.size()[-2]))

        if eea is not None or aa is not None:
            if eea is not None:
                train_input_b2 = eea(train_input_b2, use_Q_table=True)
            else:
                train_input_b2 = aa(train_input_b2)
            train_optim.zero_grad()
            with autocast(enabled=use_fp16):
                train_pred_b2 = model(train_input_b2, training=False)
                if cutmix_beta > 0 and r < cutmix_prob:
                    batch_train_loss = criterion(train_pred_b2[0], target_a.type(torch.long))
                    batch_train_loss_tmp = criterion(train_pred_b2[0], target_b.type(torch.long))
                    train_loss = torch.mean(batch_train_loss * lam + batch_train_loss_tmp * (1. - lam))
                else:
                    batch_train_loss = criterion(train_pred_b2[0], train_label_b2.type(torch.long))
                    train_loss = torch.mean(batch_train_loss)
            if eea is not None:
                eea.cal_Qmax(batch_train_loss)
                eea.update_Q_table(batch_train_loss)
        else:
            ra_deformation = torch.ones((aug_depth, N - N // 2)) * 0.3
            train_optim.zero_grad()
            with autocast(enabled=use_fp16):
                train_pred_b2 = model(train_input_b2, training=True, cos=ra_deformation)
                if cutmix_beta > 0 and r < cutmix_prob:
                    train_loss = criterion(train_pred_b2[0], target_a.type(torch.long)) * lam + criterion(train_pred_b2[0], target_b.type(torch.long)) * (1. - lam)
                else:
                    train_loss = criterion(train_pred_b2[0], train_label_b2.type(torch.long))

        scaler.scale(train_loss).backward()
        if use_fp16:
            scaler.unscale_(train_optim)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(train_optim)
        scaler.update()
        model_scheduler.step()

        # Update metrics
        train_prec1 = accuracy(train_pred_b2[0].detach(), train_label_b2.detach(), topk=(1,))
        train_loss_m.update(train_loss.detach().item(), N - N // 2)
        train_top1.update(train_prec1.detach().item(), N - N // 2)
        pred_all.append(train_pred_b2[0].detach())
        target_all.append(train_label_b2)

    if eea is not None:
        print('Bin cnts:', eea.bin_cnts)
        print('Reward:', eea.reward)
        print('Q table:', eea.Q_table)

    pred_all = torch.cat(pred_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    if pred_all[0].shape[-1] == 2:
        train_spe = specificity(pred_all, target_all)
        train_sen = sensitivity(pred_all, target_all)
        train_f1 = f1(pred_all, target_all)
        train_auc = auc(pred_all, target_all)
    else:
        train_spe = train_sen = train_f1 = train_auc = None

    return train_loss_m.avg, train_top1.avg, train_spe, train_sen, train_f1, train_auc


def infer(test_loader, model, criterion, eea, aa):
    loss_m = utils.AverageMeter()
    top1 = utils.AverageMeter()
    pred_all = []
    target_all = []

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            input, label = [x.cuda() for x in data]
            N = input.shape[0]

            if eea is not None or aa is not None:
                pred = model(input, training=False)
                batch_loss = criterion(pred[0], label.type(torch.long))
                loss = torch.mean(batch_loss)
            else:
                pred = model(input, training=False)
                loss = criterion(pred[0], label.type(torch.long))
            prec1 = accuracy(pred[0], label, topk=(1,))

            loss_m.update(loss.detach().item(), N)
            top1.update(prec1.detach().item(), N)
            pred_all.append(pred[0])
            target_all.append(label)

    pred_all = torch.cat(pred_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    if pred_all[0].shape[-1] == 2:
        val_spe = specificity(pred_all, target_all)
        val_sen = sensitivity(pred_all, target_all)
        val_f1 = f1(pred_all, target_all)
        val_auc = auc(pred_all, target_all)
    else:
        val_spe = val_sen = val_f1 = val_auc = None

    return loss_m.avg, top1.avg, val_spe, val_sen, val_f1, val_auc, pred_all, target_all


if __name__ == '__main__':
    main()