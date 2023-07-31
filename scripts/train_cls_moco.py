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


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    """
    xy = [X, U]
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def train_multi_task(cfg, epoch, model, optimizer, labeled_train_loader, use_cuda):
    global loss_value, losses
    model.train()
    timer, ec = logger.AverageMeter(), logger.AverageMeter()
    labeled_train_loader = iter(labeled_train_loader)
    for i in range(cfg.train.train_iteres):
        st = time.time()
        x, label = next(labeled_train_loader)

        if cfg.train.mixup:
            lam = np.random.beta(cfg.train.alpha, cfg.train.alpha)
            x2, label2 = next(labeled_train_loader)
            x = lam * x + (1 - lam) * x2
            if use_cuda:
                x = x.cuda()
                label, label2 = label.cuda(), label2.cuda()
            pred = model(x)
            criterion = nn.CrossEntropyLoss().cuda()
            loss_value = lam * criterion(pred, label) + (1 - lam) * criterion(pred, label2)
            losses = [loss_value]
        else:
            if use_cuda:
                x = x.cuda()
                label = label.cuda()
            y = model(x)
            for idx, loss_func in enumerate(cfg.loss.losses):
                if idx == 0:
                    loss_value = loss_func(y, label)
                    losses = [loss_value]
                else:
                    temp_loss = loss_func(y, label)
                    loss_value += temp_loss
                    losses.append(temp_loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_value.backward()
        # pdb.set_trace()
        optimizer.step()
        # pdb.set_trace()
        timer.update(time.time() - st)
        ec.update(losses[0].cpu().detach())
        msg = "epoch: {},  iter:{} / all:{}, etc:{:.2f}, ec_loss: {:.4f}". \
            format(epoch, i, cfg.train.train_iteres, time.time() - st, ec.avg)
        print(msg)
    return ec.avg


def train_MOCO_seg(cfg, epoch, model, optimizer, labeled_train_loader, use_cuda):
    global loss_value, losses
    model.train()
    criterion = cfg.loss.losses[0]
    ec_seg = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.95])).cuda()
    dice_loss = Losses.DiceLoss().cuda()
    timer, sum_loss, loss_1, loss_2, loss_3 = logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter(), \
                                        logger.AverageMeter(), logger.AverageMeter(),
    labeled_train_loader = iter(labeled_train_loader)
    for i in range(cfg.train.train_iteres):
        st = time.time()
        # if i >= 1:
        #     break
        if cfg.train.moco:
            q, mask, q_label = next(labeled_train_loader)
            q, k = q[:len(q) // 2, ...], q[len(q) // 2:, ...]
            mask_q  = mask[:len(mask) // 2, ...]
            if use_cuda:
                q, k = q.cuda(), k.cuda()
                mask_q = mask_q.cuda()

            logits, labels, pred_mask = model(q, k, q_label[0].item(), bn_shuffle=False)
            # import pdb
            # pdb.set_trace()
            # import pdb
            # pdb.set_trace()
            moco_loss = criterion(logits, labels)
            seg_ec_loss = ec_seg(pred_mask, mask_q)
            seg_dice_loss = dice_loss(pred_mask, mask_q)

            loss_value = 10*moco_loss + 0.5 * seg_ec_loss + 0.5 * seg_dice_loss
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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_value.backward()
        # pdb.set_trace()
        optimizer.step()
        # pdb.set_trace()
        timer.update(time.time() - st)

    return sum_loss.avg, loss_1.avg, loss_2.avg, loss_3.avg,


def train_MOCO_vae(cfg, epoch, model, optimizer, labeled_train_loader, use_cuda):
    global loss_value, losses
    model.train()
    criterion = cfg.loss.losses[0]
    ec_seg = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.95])).cuda()
    vae_loss = Losses.KL_LOSS().cuda()
    timer, sum_loss, loss_1, loss_2, loss_3 = logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter(), \
                                              logger.AverageMeter(), logger.AverageMeter(),
    labeled_train_loader = iter(labeled_train_loader)
    for i in range(1, cfg.train.train_iteres+1):
        # if i >= 1:
        #     break
        st = time.time()
        if cfg.train.moco:
            data, q_label = next(labeled_train_loader)
            q, k = data[:len(data)//2, ...], data[len(data)//2:, ...]
            q_label = q_label[0]
            if use_cuda:
                q = q.cuda()
                k = k.cuda()
        # if cfg.train.moco:
        #     q, k, q_label = next(labeled_train_loader)
        #     if use_cuda:
        #         q = q[0].cuda()
        #         k = k[0].cuda()

            logits, labels, recon, mu, log_var = model(q, k, q_label.item(), bn_shuffle=False)
            # import pdb
            # pdb.set_trace()
            # import pdb
            # pdb.set_trace()
            moco_loss = criterion(logits, labels)
            recons_loss, kld_loss = vae_loss(q, recon, mu, log_var)

            loss_value = moco_loss +  0.1 * recons_loss + 0.009 * kld_loss
            # import pdb
            # pdb.set_trace()
            # print(loss_value.item())
            losses = [moco_loss.item(), recons_loss.item(), kld_loss.item()]
            # import pdb
            # pdb.set_trace()
            loss_1.update(moco_loss.item())
            loss_2.update(recons_loss.item()/10)
            loss_3.update(kld_loss.item()*0.009)
            sum_loss.update(loss_value.item())
            msg = "epoch: {},  iter:{} / all:{}, ETC:{:.2f},  total: {:.4f}, moco_loss: {:.4f}, recons_loss: {:.4f}, " \
                  "kld_loss: {:.4f}".format(epoch, i,  cfg.train.train_iteres, time.time()-st,sum_loss.avg, loss_1.avg,
                                            loss_2.avg, loss_3.avg)
            print(msg)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        timer.update(time.time() - st)
    return sum_loss.avg, loss_1.avg, loss_2.avg, loss_3.avg,


def train_MOCO_raw(cfg, epoch, model, optimizer, labeled_train_loader, use_cuda):
    global loss_value, losses
    model.train()
    criterion = cfg.loss.losses[0]
    timer, sum_loss, loss_1 = logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter()
    labeled_train_loader = iter(labeled_train_loader)
    for i in range(1, cfg.train.train_iteres+1):
        # if i >= 1:
        #     break
        st = time.time()
        if cfg.train.moco:
            data, q_label = next(labeled_train_loader)
            q, k = data[:len(data)//2, ...], data[len(data)//2:, ...]
            q_label = q_label[0]
            if use_cuda:
                q = q.cuda()
                k = k.cuda()
        # if cfg.train.moco:
        #     q, k, q_label = next(labeled_train_loader)
        #     if use_cuda:
        #         q = q[0].cuda()
        #         k = k[0].cuda()

            logits, labels = model(q, k, q_label.item(), bn_shuffle=False)
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
            msg = "epoch: {},  iter:{} / all:{}, ETC:{:.2f},  total: {:.4f}, moco_loss: {:.4f}".format(epoch, i,  cfg.train.train_iteres, time.time()-st,sum_loss.avg, loss_1.avg)
            print(msg)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        timer.update(time.time() - st)
    return sum_loss.avg, loss_1.avg



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
        elif cfg.net.name == "moco_resnet50_seg":
            img,  mask, label,_ = data

            if use_cuda:
                img = img.cuda()
            logits, labels, pred = model(img, img, label.item(), val=True, bn_shuffle=True)
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

        elif cfg.net.name == "moco_resnet50_raw":
            img, label, _ = data
            if use_cuda:
                img = img.cuda()
            logit, target = model(img, img, label.item(), val=True, bn_shuffle=False)
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
        img, _, label, path = data
        if idx == 0:
            file_dir, _ = os.path.split(path[0])
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
        if use_cuda:
            img = img.cuda()
        vec = model.encoder_k.get_hidden_vec(img)
        # import pdb
        # pdb.set_trace()
        # print(idx, vec.size())
        np.save(path[0]+"{}-epoch-train.npy".format(idx), vec.detach().cpu().numpy())

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
        if cfg.net.name == "moco_resnet50_vae" or cfg.net.name == "moco_resnet50_raw" or cfg.net.name == "moco_resnet50_seg" :
            img, _, label, path = data
            if use_cuda:
                img = img.cuda()
            # logit, target, recon, mu, log_var = model(img, img, label.item(), val=True)
            vec = model.encoder_k.get_hidden_vec(img)
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


# print(os.listdir(train_dir))
def acc_func(pred, y_true):
    res = 0
    for p, t in zip(pred, y_true):
        if p == t:
            res += 1
    return res/len(pred)


def cal_cls_res(label, pred, prob=None):
    p = precision_score(label, pred, average='binary')
    r = recall_score(label, pred, average='binary')
    f1score = f1_score(label, pred, average='binary')
    print("acc:", np.sum(np.array(label)==np.array(pred))/len(pred))
    print("precision_score:", p)
    print("recall_score:", r)
    print("f1_score:", f1score)
    if prob is not None:
        fpr,tpr,threshold = roc_curve(label,prob) ###计算真正率和假正率
        roc_auc = auc(fpr,tpr)
        print("roc_auc:", roc_auc)
    return np.sum(np.array(label)==np.array(pred))/len(pred), f1score


def collect_data(train_dir, postfix="epoch60.npy"):
    train_X = []
    train_Y = []
    pathes = []
    for i in os.listdir(train_dir):
        marker = i[-len(postfix):]
        if marker == postfix :
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
    train_cls_dataset, val_cls_dataset = None, None
    if cfg.net.name == "Unet":
        model = unet.Unet(cfg.net.inchannels, cfg.net.classes, drop_rate=0.2, filters=cfg.net.enc_nf)
    elif cfg.net.name == "moco_resnet50_raw":
        # model = moco.MoCo(dim=128, K=240)
        model = moco.MoCo_raw(dim=128, K=120)
        if cfg.net.pretrain:
            model = model_io.load_retrained_model_for_moco(cfg.net.pretrain, model)
        dataset = data_generator.CLS_base_MOCO_train(cfg.data.train_txt)
        my_sample = data_generator.moco_sampler(dataset, batch_size=cfg.train.batch_size, iters=cfg.train.train_iteres)
        train_cls_dataset = DataLoader(dataset, batch_sampler=my_sample, num_workers=25)
        # train_cls_dataset = data_generator.CLS_base_MOCO(cfg.data.train_txt, cfg.train.batch_size,
        #                                                  is_aug=cfg.train.aug_data, num_class=cfg.net.classes)
        val_cls_dataset = DataLoader(data_generator.VAL_CLS_moco_vae(cfg.data.val_txt,
                                                                 outdir=cfg.general.vec_save_dir+"/test",
                                                                 num_class=cfg.net.classes),
                                     batch_size=1, shuffle=False, num_workers=1)
        test_train_dataset = DataLoader(data_generator.VAL_CLS_moco_vae(cfg.data.train_txt,
                                                                 outdir=cfg.general.vec_save_dir+"/train",
                                                                 num_class=cfg.net.classes),
                                        batch_size=1, shuffle=False, num_workers=1)
    elif cfg.net.name == "moco_resnet50_seg":
        model = moco.MoCo_seg(dim=128, K=cfg.net.mem_bank)
        if cfg.net.pretrain:
            model = model_io.load_retrained_model_for_moco(cfg.net.pretrain, model)
        dataset = data_generator.CLS_base_MOCO_train_seg(cfg.data.train_txt)
        my_sample = data_generator.moco_sampler(dataset, batch_size=cfg.train.batch_size, iters=cfg.train.train_iteres)
        train_cls_dataset = DataLoader(dataset, batch_sampler=my_sample, num_workers=32)
        val_cls_dataset = DataLoader(data_generator.VAL_CLS_by_seg(cfg.data.val_txt,
                                                                   cfg.general.vec_save_dir+"/test",
                                                                 num_class=cfg.net.classes),
                                     batch_size=1, shuffle=False)
        test_train_dataset = DataLoader(data_generator.VAL_CLS_by_seg(cfg.data.train_txt,
                                                                      cfg.general.vec_save_dir+"/train",
                                                                 num_class=cfg.net.classes),
                                        batch_size=1, shuffle=False)
    elif cfg.net.name == "moco_resnet50_vae":
        model = moco.MoCo_vae(dim=128, K=120)
        # if cfg.net.pretrain:
        #     model = model_io.load_retrained_model(cfg.net.pretrain, model)
        dataset = data_generator.CLS_base_MOCO_train(cfg.data.train_txt)
        my_sample = data_generator.moco_sampler(dataset, batch_size=cfg.train.batch_size, iters=cfg.train.train_iteres)
        train_cls_dataset = DataLoader(dataset, batch_sampler=my_sample, num_workers=25)
        # train_cls_dataset = data_generator.CLS_base_MOCO(cfg.data.train_txt, cfg.train.batch_size,
        #                                                  is_aug=cfg.train.aug_data, num_class=cfg.net.classes)
        val_cls_dataset = DataLoader(data_generator.VAL_CLS_moco_vae(cfg.data.val_txt,
                                                                 outdir=cfg.general.vec_save_dir+"/test",
                                                                 num_class=cfg.net.classes),
                                     batch_size=1, shuffle=False, num_workers=1)
        test_train_dataset = DataLoader(data_generator.VAL_CLS_moco_vae(cfg.data.train_txt,
                                                                 outdir=cfg.general.vec_save_dir+"/train",
                                                                 num_class=cfg.net.classes),
                                        batch_size=1, shuffle=False, num_workers=1)
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

    value_save = ["epoch", "learning rate", "train_total", "val_ec", "val_acc", "val_dice"]
    log = logger.Logger_csv(value_save, cfg.logger.save_path, cfg.logger.save_name)


    if cfg.train.isTrue:
        temp = 0
        for epoch in range(cfg.train.start_epoch, cfg.train.epochs + 1):
            if cfg.net.name == "moco_resnet50_vae":
                loss_sum, loss1, loss2, loss3 = train_MOCO_vae(cfg, epoch, model, optimizer, train_cls_dataset, use_cuda)
                writer.add_scalar("train/Sum_Loss", loss_sum, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/moco loss", loss1, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/recon_loss", loss2, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/kld_loss", loss3, epoch * cfg.train.train_iteres)
                with torch.no_grad():
                    infoNce, Recon_loss, KL_loss = val_moco(cfg, model, val_cls_dataset, use_cuda)
                    acc_value, f1_value = test_MOCO(cfg, epoch, model, test_train_dataset, val_cls_dataset, use_cuda)
                writer.add_scalar("val/val_infoNce", infoNce, epoch * cfg.train.train_iteres)
                writer.add_scalar("val/val_Recon_loss", Recon_loss, epoch * cfg.train.train_iteres)
                writer.add_scalar("val/val_KL_loss", KL_loss, epoch * cfg.train.train_iteres)
                writer.add_scalar("test/acc", acc_value, epoch * cfg.train.train_iteres)
                writer.add_scalar("test/f1_value", f1_value, epoch * cfg.train.train_iteres)
            elif cfg.net.name == "moco_resnet50_raw":
                loss_sum, loss1 = train_MOCO_raw(cfg, epoch, model, optimizer, train_cls_dataset, use_cuda)
                writer.add_scalar("train/Sum_Loss", loss_sum, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/moco loss", loss1, epoch * cfg.train.train_iteres)
                with torch.no_grad():
                    infoNce, Recon_loss, KL_loss = val_moco(cfg, model, val_cls_dataset, use_cuda)
                    acc_value, f1_value = test_MOCO(cfg, epoch, model, test_train_dataset, val_cls_dataset, use_cuda)
                writer.add_scalar("val/val_infoNce", infoNce, epoch * cfg.train.train_iteres)
                writer.add_scalar("test/acc", acc_value, epoch * cfg.train.train_iteres)
                writer.add_scalar("test/f1_value", f1_value, epoch * cfg.train.train_iteres)
            elif cfg.net.name == "moco_resnet50_seg":
                loss_sum, loss1, loss2, loss3 = train_MOCO_seg(cfg, epoch, model, optimizer, train_cls_dataset, use_cuda)
                writer.add_scalar("train/Sum_Loss", loss_sum, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/moco loss", loss1, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/seg_loss_ec", loss2, epoch * cfg.train.train_iteres)
                writer.add_scalar("train/seg_loss_dice", loss3, epoch * cfg.train.train_iteres)
                with torch.no_grad():
                    infoNce, Recon_loss, KL_loss = val_moco(cfg, model, val_cls_dataset, use_cuda)
                    acc_value, f1_value = test_MOCO(cfg, epoch, model, test_train_dataset, val_cls_dataset, use_cuda)
                writer.add_scalar("val/val_infoNce", infoNce, epoch * cfg.train.train_iteres)
                writer.add_scalar("val/val_HAUFF_loss", Recon_loss, epoch * cfg.train.train_iteres)
                writer.add_scalar("val/val_dice_loss", KL_loss, epoch * cfg.train.train_iteres)
                writer.add_scalar("test/acc", acc_value, epoch * cfg.train.train_iteres)
                writer.add_scalar("test/f1_value", f1_value, epoch * cfg.train.train_iteres)

            else:
                acc_value, f1_value = test_MOCO(cfg, epoch, model, test_train_dataset, val_cls_dataset, use_cuda)
                torch.save(dict(epoch=cfg.train.epochs + 1,
                                state_dict=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict()),
                           f=os.path.join(cfg.general.model_path, "final_model.pth"))
                # loss_sum, loss1, loss2, loss3 = train_MOCO_seg(cfg, epoch, model, optimizer, train_cls_dataset, use_cuda)
                # writer.add_scalar("train/Sum_Loss", loss_sum, epoch * cfg.train.train_iteres)
                # writer.add_scalar("train/moco loss", loss1, epoch * cfg.train.train_iteres)
                # writer.add_scalar("train/seg_loss_ec", loss2, epoch * cfg.train.train_iteres)
                # writer.add_scalar("train/seg_loss_dice", loss3, epoch * cfg.train.train_iteres)

            # log.update({"epoch": epoch, "learning rate": optimizer.param_groups[0]['lr'],
            #             "train_total": loss_sum})

            scheduler.step()
            print("learning rate", optimizer.param_groups[0]['lr'])
            if epoch % cfg.train.save_epoch == 0:
                if not os.path.isdir(cfg.general.model_path):
                    os.makedirs(cfg.general.model_path)
                # acc_value, f1_value = test_MOCO(epoch, model, val_cls_dataset, test_train_dataset, use_cuda)
                torch.save(dict(epoch=epoch,
                                state_dict=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict()),
                           f=os.path.join(cfg.general.model_path, str(epoch) + "_model.pth"))

    else:
        acc_value, f1_value = test_MOCO(cfg, -1, model, test_train_dataset, val_cls_dataset, use_cuda)



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
