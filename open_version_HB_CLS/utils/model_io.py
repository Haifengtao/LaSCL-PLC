#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model_io.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/14 16:31   Bot Zhao      1.0         None
"""

# import lib
import os
import torch


# TODO remove dependency to args
def reload_ckpt(path, model, optimizer, scheduler, use_cuda):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except Exception:
            net_dict = model.state_dict()
            # net_dict.update(checkpoint['state_dict'])
            for i in checkpoint['state_dict']:
                if i in net_dict:
                    if net_dict[i].size() == checkpoint['state_dict'][i].size():
                        net_dict[i] = checkpoint['state_dict'][i]
                        print(i)
                    else:
                        continue
            model.load_state_dict(net_dict)


        if use_cuda:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 因为optimizer加载参数时,tensor默认在CPU上
            # 故需将所有的tensor都放到cuda,
            # 否则: 在optimizer.step()处报错：
            # RuntimeError: expected device cpu but got device cuda:0
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            model.cuda()
        else:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception:
                print(Exception)
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("started with:", checkpoint['epoch']))
        return start_epoch
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(path))


def reload_ckpt_for_eval(path, model, use_cuda):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(Exception)
            net_dict = model.state_dict()
            # net_dict.update(checkpoint['state_dict'])
            for i in checkpoint['state_dict']:
                if i in net_dict:
                    if net_dict[i].size() == checkpoint['state_dict'][i].size():
                        net_dict[i] = checkpoint['state_dict'][i]
                        print(i)
                    else:
                        continue
            model.load_state_dict(net_dict)


        if use_cuda:
            model.cuda()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("started with:", checkpoint['epoch']))
        return start_epoch, model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(path))


def load_retrained_model(pth_dir, model):
    net_dict = {}
    print('loading pretrained model {}'.format(pth_dir))
    pretrain = torch.load(pth_dir)
    print(list(pretrain['state_dict'].keys())[:10])
    print(list(net_dict.keys())[:10])

    for i in pretrain['state_dict']:
        if i[7:] in model.state_dict():
            net_dict[i[7:]] = pretrain['state_dict'][i]
            print(i)
    import pdb
    pdb.set_trace()
    #
    # pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict.keys()}
    # print("updated: {}".format(pretrain_dict.keys()))

    temp = model.state_dict()
    temp.update(net_dict)
    model.load_state_dict(temp)
    # import pdb
    # pdb.set_trace()
    return model


def load_retrained_model_for_swinunet(pth_dir, model):
    net_dict = {}
    print('loading pretrained model {}'.format(pth_dir))
    pretrain = torch.load(pth_dir)
    print(list(pretrain['state_dict'].keys())[:10])
    print(list(model.state_dict().keys())[:10])

    for i in pretrain['state_dict']:
        if i in model.state_dict() and pretrain['state_dict'][i].size()==model.state_dict()[i].size():
            net_dict[i] = pretrain['state_dict'][i]
            print(i)
    # import pdb
    # pdb.set_trace()
    #
    # pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict.keys()}
    # print("updated: {}".format(pretrain_dict.keys()))

    temp = model.state_dict()
    temp.update(net_dict)
    model.load_state_dict(temp)
    # import pdb
    # pdb.set_trace()
    return model


def load_retrained_model_for_moco(pth_dir, model):
    net_dict = model.state_dict()
    print('loading pretrained model {}'.format(pth_dir))
    pretrain = torch.load(pth_dir)
    print(list(pretrain['state_dict'].keys())[:10])
    print(list(net_dict.keys())[:10])
    # import pdb
    # pdb.set_trace()
    pretrain_dict = {k.replace('module.','encoder_q.'): v for k, v in pretrain['state_dict'].items() if
                     k.replace('module.','encoder_q.') in net_dict.keys()}
    print("updated: {}".format(pretrain_dict.keys()))

    net_dict.update(pretrain_dict)

    pretrain_dict= []
    pretrain_dict = {k.replace('module.','encoder_k.'): v for k, v in pretrain['state_dict'].items() if
                     k.replace('module.', 'encoder_k.') in net_dict.keys()}
    print("updated: {}".format(pretrain_dict.keys()))
    net_dict.update(pretrain_dict)

    model.load_state_dict(net_dict)
    # import pdb
    # pdb.set_trace()
    return model