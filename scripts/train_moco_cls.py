#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_cls.py
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/2 19:16   Bot Zhao      1.0         None
"""

# import lib
import time
# from progress.bar import Bar as Bar
import torch
from torch import optim, nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append("./")
# sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import misc
from nets import Losses
from utils import data_generator
from nets import unet, resnet, moco
from utils import file_io
from utils import logger
from utils import model_io

nn.Conv1d
def train_MOCO_seg(cfg, epoch, model, optimizer, labeled_train_loader, use_cuda):
    global loss_value, losses
    model.train()
    criterion = cfg.loss.losses[0]
    ec_seg = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.95])).cuda()
    # ec_cls = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7])).cuda()
    ec_cls = Losses.FocalLoss()
    dice_loss = Losses.DiceLoss().cuda()
    timer, sum_loss, loss_1, loss_2, loss_3, loss_4 = logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter(), \
                                                      logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter(),
    labeled_train_loader = iter(labeled_train_loader)
    try:
        lambda1 = cfg.train.lambda1
        lambda2 = cfg.train.lambda2
        lambda3 = cfg.train.lambda3
    except:
        lambda1 = 1
        lambda2 = 0.5
        lambda3 = 0.5

    print([lambda1, lambda2, lambda3])
    for i in range(cfg.train.train_iteres):
        st = time.time()
        # if i > 1:
        #     break
        if cfg.net.name == "MoCo_seg_cls":
            q, mask, q_label, age, sex = next(labeled_train_loader)
            q, k = q[:len(q) // 2, ...], q[len(q) // 2:, ...]
            age, sex, q_label = age[:len(age) // 2, ...], sex[:len(sex) // 2, ...], q_label[:len(q_label) // 2, ...]
            mask_q = mask[:len(mask) // 2, ...]
            if use_cuda:
                q, k, age, sex = q.cuda(), k.cuda(), age.cuda(), sex.cuda()
                q_label = q_label.cuda()
                mask_q = mask_q.cuda()
            if epoch <= cfg.train.train_cl_epochs:
                logits, labels, pred_mask = model(q, k, q_label[0].item(), age, sex, train_cl=True, bn_shuffle=False)
                # import pdb
                # pdb.set_trace()
                # import pdb
                # pdb.set_trace()
                moco_loss = criterion(logits, labels)
                seg_ec_loss = ec_seg(pred_mask, mask_q)
                seg_dice_loss = dice_loss(pred_mask, mask_q)

                loss_value = lambda1 * moco_loss + lambda2 * seg_ec_loss + lambda3 * seg_dice_loss

                losses = [moco_loss.item(), seg_ec_loss.item(), seg_dice_loss.item()]
                # import pdb
                # pdb.set_trace()
                loss_1.update(moco_loss.item())
                loss_2.update(seg_ec_loss.item())
                loss_3.update(seg_dice_loss.item())
                sum_loss.update(loss_value.item())
                msg = "epoch: {},  iter:{} / all:{},  total: {:.4f}, moco_loss: {:.4f}, seg_ec_lodd: {:.4f}, " \
                      "seg_dice_loss: {:.4f}".format(epoch, i, cfg.train.train_iteres, sum_loss.avg,
                                                     loss_1.avg, loss_2.avg, loss_3.avg)
                print(msg)
            else:
                pred = model(q, k, q_label[0].item(), age, sex, train_cl=False, bn_shuffle=False)
                loss_value = ec_cls(pred, q_label)
                loss_4.update(loss_value.item())
                msg = "epoch: {},  iter:{} / all:{},  total: {:.4f}, cls_ec: {:.4f}".format(epoch, i,
                                                                                            cfg.train.train_iteres,
                                                                                            loss_4.avg,
                                                                                            loss_4.avg)
                print(msg)
        # compute gradient and do SGD step
        elif cfg.net.name == "MoCo_cls":
            q, mask, q_label, _, _ = next(labeled_train_loader)
            q, k = q[:len(q) // 2, ...], q[len(q) // 2:, ...]
            q_label = q_label[:len(q_label) // 2, ...]
            if use_cuda:
                q, k = q.cuda(), k.cuda()
                q_label = q_label.cuda()
            if epoch <= cfg.train.train_cl_epochs:
                logits, labels = model(q, k, q_label[0].item(), train_cl=True, bn_shuffle=False)
                # import pdb
                # pdb.set_trace()
                # import pdb
                # pdb.set_trace()
                moco_loss = criterion(logits, labels)

                loss_value = moco_loss
                losses = [moco_loss.item()]
                # import pdb
                # pdb.set_trace()
                loss_1.update(moco_loss.item())
                sum_loss.update(loss_value.item())
                msg = "epoch: {},  iter:{} / all:{},  total: {:.4f}, moco_loss: {:.4f}". \
                    format(epoch, i, cfg.train.train_iteres, sum_loss.avg,
                           loss_1.avg, )
                print(msg)
        optimizer.zero_grad()
        loss_value.backward()
        # pdb.set_trace()
        optimizer.step()
        # pdb.set_trace()
        timer.update(time.time() - st)

    return sum_loss.avg, loss_1.avg, loss_2.avg, loss_3.avg, loss_4.avg,


def acc(pred, y_true):
    res = 0
    for p, t in zip(pred, y_true):
        if p == t:
            res += 1
    return res / len(pred)


def val_moco(cfg, model, val_dataset, use_cuda):
    # implemented
    model.eval()
    if use_cuda:
        model.cuda()
    infoNce = logger.AverageMeter()
    Recon_loss, KL_loss = logger.AverageMeter(), logger.AverageMeter()
    loss = nn.CrossEntropyLoss().cuda()
    vae_loss = Losses.KL_LOSS()
    for idx, data in enumerate(val_dataset):
        # import pdb
        # pdb.set_trace()
        # if idx > 1:
        #     break
        if cfg.net.name == "moco_resnet50_vae":
            img, label, _ = data
            if use_cuda:
                img = img.cuda()
            logits, labels, recon, mu, log_var = model(img, img, label.item(), val=True, bn_shuffle=False)
            recons_loss, kld_loss = vae_loss(img, recon, mu, log_var)
            loss_value = loss(logits, labels)
            # import pdb
            # pdb.set_trace()
            Recon_loss.update(recons_loss.item())
            KL_loss.update(kld_loss.item())
            infoNce.update(loss_value.item())
        elif cfg.net.name == "MoCo_seg_cls" or cfg.net.name == "moco_resnet50_seg":
            img, mask, label, age, sex, _ = data

            if use_cuda:
                img = img.cuda()
                age, sex = age.cuda(), sex.cuda()
            logits, labels, pred = model(img, img, label.item(), age, sex, train_cl=True, val=True, bn_shuffle=True)
            loss_value = loss(logits, labels)
            # import pdb
            # pdb.set_trace()
            infoNce.update(loss_value.item())
            pred = torch.argmax(pred, dim=1)
            dice_value = Losses.calculate_dice_coefficient(mask.squeeze().numpy(), pred.cpu().squeeze().numpy(), 1)
            iou_value = Losses.calculate_iou(mask.squeeze().numpy(), pred.cpu().squeeze().numpy(), 1)
            # hauff_value = Losses.computeQualityMeasures(mask.squeeze().numpy(), pred.cpu().squeeze().numpy())
            Recon_loss.update(iou_value)
            KL_loss.update(dice_value)
            # print("infoNce loss {}, dice:{}, hauff:{}".format(infoNce.avg, KL_loss.avg, Recon_loss.avg))

        elif cfg.net.name == "moco_resnet50_raw" or cfg.net.name == "MoCo_cls":
            # img, label, _ = data
            if use_cuda:
                img = data[0].cuda()
            logit, target = model(img, img, data[2].item(), val=True, bn_shuffle=False)
            loss_value = loss(logit, target)
            # import pdb
            # pdb.set_trace()
            infoNce.update(loss_value.item())
        else:
            raise Exception("check your config")

    print("infoNce loss {}, dice:{}, IOU:{}".format(infoNce.avg, KL_loss.avg, Recon_loss.avg))
    return infoNce.avg, Recon_loss.avg, KL_loss.avg

def test_MOCO(cfg, epoch, model, train_dataset, val_dataset, use_cuda):
    # implemented
    model.eval()
    if use_cuda:
        model.cuda()

    for idx, data in enumerate(train_dataset):
        img, _, label, age, sex, path = data
        if idx == 0:
            file_dir, _ = os.path.split(path[0])
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
        if use_cuda:
            img = img.cuda()
        vec = model.encoder_q.get_hidden_vec(img)
        # import pdb
        # pdb.set_trace()
        # print(idx, vec.size())
        np.save(path[0] + "{}-epoch-train.npy".format(idx), vec.detach().cpu().numpy())

    for idx, data in enumerate(val_dataset):
        # if cfg.train.moco:
        #     img, tumor, label, path = data
        #     if idx == 0:
        #         file_dir, _ = os.path.split(path[0])
        #         if not os.path.isdir(file_dir):
        #             os.makedirs(file_dir)
        #     if use_cuda:
        #         img = img.cuda()
        #         tumor = tumor.cuda()
        #     # logit, target, seg = model(img, img, label.item(), val=True)
        #     vec = model.encoder_k.get_hidden_vec(img)
        #     # import pdb
        #     # pdb.set_trace()
        #     np.save(path[0]+"-epoch"+str(epoch), vec.detach().cpu().numpy())
        dice_values = []
        if cfg.net.name == "MoCo_seg_cls" or cfg.net.name == "moco_resnet50_raw" or cfg.net.name == "MoCo_cls" \
                or cfg.net.name == "moco_resnet50_seg":
            img, mask, label, age, sex, path = data
            if use_cuda:
                img = img.cuda()
            # logit, target, recon, mu, log_var = model(img, img, label.item(), val=True)
            logits, labels, pred_mask = model(img, img, label[0], age, sex, train_cl=True, bn_shuffle=False, val=True)

            mask_arr = mask.squeeze().detach().numpy().astype('uint8')
            pred_mask_arr = torch.max(pred_mask, 1)[1].cpu().squeeze().detach().numpy().astype('uint8')
            dice_value = 2 * np.sum(mask_arr * pred_mask_arr) / (np.sum(mask_arr) + np.sum(pred_mask_arr) + 1)
            dice_values.append(dice_value)
            # print(dice_value)
            # if dice_value <= 0.01:
            #     file_io.save_nii_array(mask_arr, './test_gt.nii', )
            #     file_io.save_nii_array(pred_mask_arr, './test_pred.nii', )
                # break

            # import pdb
            # pdb.set_trace()
            # dice_value =
            vec = model.encoder_q.get_hidden_vec(img)
            if idx == 0:
                file_dir, _ = os.path.split(path[0])
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
            np.save(path[0] + "{}-test.npy".format(idx), vec.detach().cpu().numpy())

        else:
            raise Exception("check your config")

    # print("TEST: infoNce loss {}, dice: {}".format(infoNce.avg, dices.avg))
    train_dir = cfg.general.vec_save_dir + '/train'
    test_dir = cfg.general.vec_save_dir + '/test'
    train_X, train_Y, train_pathes = collect_data(train_dir, "train.npy")
    test_X, test_Y, test_pathes = collect_data(test_dir, "test.npy")
    xgb1 = XGBClassifier()
    xgb1.fit(train_X, train_Y)
    pred_prob = xgb1.predict_proba(test_X)
    pred_label = xgb1.predict(test_X)
    acc_value, f1_value = cal_cls_res(test_Y, pred_label, pred_prob[:, 1])
    return acc_value, f1_value, np.mean(dice_values)


def test_MOCO_train(cfg, epoch, model, train_dataset, val_dataset, use_cuda):
    # implemented
    model.eval()
    if use_cuda:
        model.cuda()

    for idx, data in enumerate(train_dataset):
        img, _, label, age, sex, path = data
        if idx == 0:
            file_dir, _ = os.path.split(path[0])
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
        if use_cuda:
            img = img.cuda()
        vec = model.encoder_q.get_hidden_vec(img)
        # import pdb
        # pdb.set_trace()
        # print(idx, vec.size())
        np.save(path[0] + "{}-epoch-train.npy".format(idx), vec.detach().cpu().numpy())

    for idx, data in enumerate(val_dataset):
        # if cfg.train.moco:
        #     img, tumor, label, path = data
        #     if idx == 0:
        #         file_dir, _ = os.path.split(path[0])
        #         if not os.path.isdir(file_dir):
        #             os.makedirs(file_dir)
        #     if use_cuda:
        #         img = img.cuda()
        #         tumor = tumor.cuda()
        #     # logit, target, seg = model(img, img, label.item(), val=True)
        #     vec = model.encoder_k.get_hidden_vec(img)
        #     # import pdb
        #     # pdb.set_trace()
        #     np.save(path[0]+"-epoch"+str(epoch), vec.detach().cpu().numpy())
        if cfg.net.name == "MoCo_seg_cls" or cfg.net.name == "moco_resnet50_raw" or cfg.net.name == "MoCo_cls" \
                or cfg.net.name == "moco_resnet50_seg":
            img, mask, label, age, sex, path = data
            if use_cuda:
                img = img.cuda()
            # logit, target, recon, mu, log_var = model(img, img, label.item(), val=True)
            logits, labels, pred_mask = model(img, img, label[0], age, sex, train_cl=True, bn_shuffle=False)

            mask_arr = mask.squeeze().detach().numpy().astype('uint8')
            pred_mask_arr = torch.max(pred_mask, 1)[1].cpu().squeeze().detach().numpy().astype('uint8')
            dice_value = 2 * np.sum(mask_arr * pred_mask_arr) / (np.sum(mask_arr) + np.sum(pred_mask_arr) + 1)
            print(dice_value)
            # if dice_value <= 0.01:
            #     file_io.save_nii_array(mask_arr, './test_gt.nii', )
            #     file_io.save_nii_array(pred_mask_arr, './test_pred.nii', )
            #     break

            # import pdb
            # pdb.set_trace()
            # dice_value =
            vec = model.encoder_q.get_hidden_vec(img)
            if idx == 0:
                file_dir, _ = os.path.split(path[0])
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
            np.save(path[0] + "{}-test.npy".format(idx), vec.detach().cpu().numpy())

        else:
            raise Exception("check your config")

    # print("TEST: infoNce loss {}, dice: {}".format(infoNce.avg, dices.avg))
    train_dir = cfg.general.vec_save_dir + '/train'
    test_dir = cfg.general.vec_save_dir + '/test'
    train_X, train_Y, train_pathes = collect_data(train_dir, "train.npy")
    test_X, test_Y, test_pathes = collect_data(test_dir, "test.npy")
    xgb1 = XGBClassifier()
    xgb1.fit(train_X, train_Y)
    pred_prob = xgb1.predict_proba(test_X)
    pred_label = xgb1.predict(test_X)
    acc_value, f1_value = cal_cls_res(test_Y, pred_label, pred_prob[:, 1])
    return acc_value, f1_value


def test_MOCO_cls(model, val_dataset, use_cuda):
    # implemented
    model.eval()
    if use_cuda:
        model.cuda()
    pred_label = []
    pred_prob = []
    labels = []
    ec_cls = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 0.6])).cuda()
    loss_4 = logger.AverageMeter()
    for idx, data in enumerate(val_dataset):
        # if cfg.train.moco:
        #     img, tumor, label, path = data
        #     if idx == 0:
        #         file_dir, _ = os.path.split(path[0])
        #         if not os.path.isdir(file_dir):
        #             os.makedirs(file_dir)
        #     if use_cuda:
        #         img = img.cuda()
        #         tumor = tumor.cuda()
        #     # logit, target, seg = model(img, img, label.item(), val=True)
        #     vec = model.encoder_k.get_hidden_vec(img)
        #     # import pdb
        #     # pdb.set_trace()
        #     np.save(path[0]+"-epoch"+str(epoch), vec.detach().cpu().numpy())
        img, _, label, age, sex, path = data
        if use_cuda:
            img, label = img.cuda(), label.cuda()
            age, sex = age.cuda(), sex.cuda()
        # logit, target, recon, mu, log_var = model(img, img, label.item(), val=True)
        pred_temp = model(img, img, -1, age, sex, train_cl=False, bn_shuffle=False)
        loss_value = ec_cls(pred_temp, label)
        loss_4.update(loss_value.item())
        pred_temp = nn.functional.softmax(pred_temp, dim=1)
        pred_l = torch.max(pred_temp, dim=1)[1]
        pred_label += [pred_l.item()]
        pred_prob += [pred_temp.cpu().detach().squeeze().numpy()]
        labels += [label.item()]
    acc_value, f1_value = cal_cls_res(labels, pred_label, np.array(pred_prob)[:, 1])
    return acc_value, f1_value, loss_4.avg


# print(os.listdir(train_dir))
def acc_func(pred, y_true):
    res = 0
    for p, t in zip(pred, y_true):
        if p == t:
            res += 1
    return res / len(pred)


def cal_cls_res(label, pred, prob=None):
    p = precision_score(label, pred, average='binary')
    r = recall_score(label, pred, average='binary')
    f1score = f1_score(label, pred, average='binary')
    print("acc:", np.sum(np.array(label) == np.array(pred)) / len(pred))
    print("precision_score:", p)
    print("recall_score:", r)
    print("f1_score:", f1score)
    if prob is not None:
        fpr, tpr, threshold = roc_curve(label, prob)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)
        print("roc_auc:", roc_auc)
    return np.sum(np.array(label) == np.array(pred)) / len(pred), f1score


def collect_data(train_dir, postfix="epoch60.npy"):
    train_X = []
    train_Y = []
    pathes = []
    for i in os.listdir(train_dir):
        marker = i[-len(postfix):]
        if marker == postfix:
            pathes.append(i)
            train_X.append(np.load(os.path.join(train_dir, i)))
            if "hemangioblastoma" in i:
                train_Y.append(1)
            else:
                train_Y.append(0)
    train_X = np.array(train_X).squeeze()
    # import pdb
    # pdb.set_trace()
    train_X = np.array(nn.functional.normalize(torch.from_numpy(train_X), dim=1))
    return train_X, np.array(train_Y).squeeze(), pathes


def main(cfg):
    train_cls_dataset, val_cls_dataset, train_cl_dataset = None, None, None
    if cfg.net.name == "Unet":
        model = unet.Unet(cfg.net.inchannels, cfg.net.classes, drop_rate=0.2, filters=cfg.net.enc_nf)
    elif cfg.net.name == "MoCo_seg_cls":
        model = moco.MoCo_seg_cls(dim=128, K=cfg.net.mem_bank)
        if cfg.net.pretrain:
            model = model_io.load_retrained_model_for_moco(cfg.net.pretrain, model)
        dataset = data_generator.CLS_base_MOCO_train_seg_v2(cfg.data.train_txt)
        my_sample = data_generator.moco_sampler(dataset, batch_size=cfg.train.batch_size, iters=cfg.train.train_iteres)
        train_cl_dataset = DataLoader(dataset, batch_sampler=my_sample, num_workers=32)
        my_sample2 = data_generator.moco_sampler2(dataset, batch_size=cfg.train.batch_size // 2,
                                                  iters=cfg.train.train_iteres)
        train_cls_dataset = DataLoader(dataset, batch_sampler=my_sample2, num_workers=32)
        val_cls_dataset = DataLoader(data_generator.VAL_CLS_by_seg_v2(cfg.data.val_txt,
                                                                      cfg.general.vec_save_dir + "/test",
                                                                      num_class=cfg.net.classes),
                                     batch_size=1, shuffle=False)
        test_train_dataset = DataLoader(data_generator.VAL_CLS_by_seg_v2(cfg.data.train_txt,
                                                                         cfg.general.vec_save_dir + "/train",
                                                                         num_class=cfg.net.classes),
                                        batch_size=1, shuffle=False)
    elif cfg.net.name == "MoCo_cls":
        model = moco.MoCo_raw(dim=128, K=cfg.net.mem_bank)
        if cfg.net.pretrain:
            model = model_io.load_retrained_model_for_moco(cfg.net.pretrain, model)
        dataset = data_generator.CLS_base_MOCO_train_seg_v2(cfg.data.train_txt)
        my_sample = data_generator.moco_sampler(dataset, batch_size=cfg.train.batch_size, iters=cfg.train.train_iteres)
        train_cl_dataset = DataLoader(dataset, batch_sampler=my_sample, num_workers=32)
        my_sample2 = data_generator.moco_sampler2(dataset, batch_size=cfg.train.batch_size // 2,
                                                  iters=cfg.train.train_iteres)
        train_cls_dataset = DataLoader(dataset, batch_sampler=my_sample2, num_workers=32)
        val_cls_dataset = DataLoader(data_generator.VAL_CLS_by_seg_v2(cfg.data.val_txt,
                                                                      cfg.general.vec_save_dir + "/test",
                                                                      num_class=cfg.net.classes),
                                     batch_size=1, shuffle=False)
        test_train_dataset = DataLoader(data_generator.VAL_CLS_by_seg_v2(cfg.data.train_txt,
                                                                         cfg.general.vec_save_dir + "/train",
                                                                         num_class=cfg.net.classes),
                                        batch_size=1, shuffle=False)
    else:
        raise Exception("We have not implemented this model {}".format(cfg.net.name))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu

    # train_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.train_txt), batch_size=cfg.train.batch_size,
    #                                shuffle=True)
    # val_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.val_txt), batch_size=1, shuffle=False)

    writer = SummaryWriter(cfg.general.log_path)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0, )

    if torch.cuda.is_available():
        use_cuda = True
        model = model.cuda()
    else:
        use_cuda = False

    if cfg.train.warmRestart:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cur_iter: (1 + cur_iter) / (
                cfg.train.train_iteres * cfg.train.warm))

    if cfg.train.load_model is not None:
        start_epoch = model_io.reload_ckpt(cfg.train.load_model, model, optimizer, scheduler=scheduler,
                                           use_cuda=use_cuda)
        cfg.train.start_epoch = start_epoch

    # for params in optimizer.param_groups:
    #     # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
    #     params['lr'] = 0.0001
    value_save = ["epoch", "learning rate", "train_total", "val_ec", "val_acc", "val_dice"]
    log = logger.Logger_csv(value_save, cfg.logger.save_path, cfg.logger.save_name)

    if cfg.train.isTrue:
        temp = 0
        for epoch in range(cfg.train.start_epoch + 1, cfg.train.epochs + 1):
            if cfg.net.name == "MoCo_seg_cls" or cfg.net.name == "MoCo_cls":

                # training
                if epoch <= cfg.train.train_cl_epochs:
                    loss_sum, loss1, loss2, loss3, loss4 = train_MOCO_seg(cfg, epoch, model, optimizer,
                                                                          train_cl_dataset, use_cuda)
                else:
                    loss_sum, loss1, loss2, loss3, loss4 = train_MOCO_seg(cfg, epoch, model, optimizer,
                                                                          train_cls_dataset,
                                                                          use_cuda)
                writer.add_scalar("train/Sum_Loss", loss_sum, epoch)
                writer.add_scalar("train/moco loss", loss1, epoch)
                writer.add_scalar("train/seg_loss_ec", loss2, epoch)
                writer.add_scalar("train/seg_loss_dice", loss3, epoch)
                writer.add_scalar("train/cls_loss_ec", loss4, epoch)

                # testing
                with torch.no_grad():
                    infoNce, Recon_loss, KL_loss, ec_cls = 0, 0, 0, 0
                    if epoch < cfg.train.train_cl_epochs:
                        infoNce, Recon_loss, KL_loss = val_moco(cfg, model, val_cls_dataset, use_cuda)
                        writer.add_scalar("val/val_infoNce", infoNce, epoch)
                        writer.add_scalar("val/val_HAUFF_loss", Recon_loss, epoch)
                        writer.add_scalar("val/val_dice_loss", KL_loss, epoch)
                        writer.add_scalar("val/cls_loss_ec", ec_cls, epoch)
                        if epoch % 10 == 9 or epoch % 10 == 8 or epoch % 10 == 0:
                            acc_value, f1_value, dice_val = test_MOCO(cfg, epoch, model, test_train_dataset, val_cls_dataset,
                                                            use_cuda)
                            print(acc_value, f1_value, dice_val)
                            writer.add_scalar("test/acc", acc_value, epoch)
                            writer.add_scalar("test/f1_value", f1_value, epoch)
                            writer.add_scalar("test/dice", dice_val, epoch)
                            torch.save(dict(epoch=epoch,
                                            state_dict=model.state_dict(),
                                            optimizer=optimizer.state_dict(),
                                            scheduler=scheduler.state_dict()),
                                       f=os.path.join(cfg.general.model_path, str(epoch) + "_model.pth"))
                    else:
                        acc_value, f1_value, ec_cls = test_MOCO_cls(model, val_cls_dataset, use_cuda)
                        writer.add_scalar("test/acc", acc_value, epoch)
                        writer.add_scalar("test/f1_value", f1_value, epoch)


            else:
                acc_value, f1_value = test_MOCO(cfg, epoch, model, test_train_dataset, val_cls_dataset, use_cuda)
                # loss_sum, loss1, loss2, loss3 = train_MOCO_seg(cfg, epoch, model, optimizer, train_cls_dataset, use_cuda)
                # writer.add_scalar("train/Sum_Loss", loss_sum, epoch * cfg.train.train_iteres)
                # writer.add_scalar("train/moco loss", loss1, epoch * cfg.train.train_iteres)
                # writer.add_scalar("train/seg_loss_ec", loss2, epoch * cfg.train.train_iteres)
                # writer.add_scalar("train/seg_loss_dice", loss3, epoch * cfg.train.train_iteres)

            # log.update({"epoch": epoch, "learning rate": optimizer.param_groups[0]['lr'],
            #             "train_total": loss_sum})

            scheduler.step()
            print("learning rate", optimizer.param_groups[0]['lr'])
            writer.add_scalar("train/learning rate", optimizer.param_groups[0]['lr'], epoch)
            if epoch % cfg.train.save_epoch == 0:
                if not os.path.isdir(cfg.general.model_path):
                    os.makedirs(cfg.general.model_path)
                # acc_value, f1_value = test_MOCO(epoch, model, val_cls_dataset, test_train_dataset, use_cuda)

            torch.save(dict(epoch=epoch,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict()),
                       f=os.path.join(cfg.general.model_path, "latest_model.pth"))

    else:
        _, model = model_io.reload_ckpt_for_eval(cfg.test.model, model, use_cuda)
        acc_value, f1_value = test_MOCO(cfg, -1, model, test_train_dataset, val_cls_dataset, use_cuda)

    torch.save(dict(epoch=cfg.train.epochs + 1,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict()),
               f=os.path.join(cfg.general.model_path, "final_model.pth"))


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Brain Training')
    parser.add_argument('-i', '--config', default=None,
                        help='model config (default: Unet)')
    arguments = parser.parse_args()
    config = file_io.load_module_from_disk(arguments.config)
    cfg = config.cfg
    # print(cfg.loss.weight)
    print(cfg.data.train_txt)
    print(cfg.train.gpu)
    main(cfg)
