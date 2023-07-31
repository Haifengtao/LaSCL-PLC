# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch, sys
import torch.nn as nn
sys.path.append("/home/zhang_istbi/zhaobt/projects/tumor_cls/")
from nets import resnet
from utils import model_io

class MoCo_seg(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=128, K=160, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_seg, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet.resnet50_seg_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)
        self.encoder_k = resnet.resnet50_seg_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)

        # self.cls_head = nn.Sequential(nn.Linear(257, 128), nn.ReLU)
        # self.encoder_q = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_q)
        # self.encoder_k = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_k)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)



        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_pos", torch.randn(dim, K))
        self.register_buffer("queue_neg", torch.randn(dim, K))
        self.queue_pos = nn.functional.normalize(self.queue_pos, dim=0)
        self.queue_neg = nn.functional.normalize(self.queue_neg, dim=0)
        self.register_buffer("queue_pos_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_neg_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pos(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_pos_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_pos[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_pos_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_neg(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_neg_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_neg[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_neg_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # import pdb
        # pdb.set_trace()
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        #         x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        #         gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x[idx_unshuffle]

    def forward(self, im_q, im_k, q_label, val=False, bn_shuffle=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, pred_1 = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if not val:
                self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            if bn_shuffle:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            else:
                k, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if q_label == 1:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_neg.clone().detach()])
            # dequeue and enqueue
            if not val:
                self._dequeue_and_enqueue_pos(k)
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_pos.clone().detach()])
            # dequeue and enqueue
            if not val:
                self._dequeue_and_enqueue_neg(k)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels, pred_1


class MoCo_seg_cls(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=128, K=160, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_seg_cls, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet.resnet50_seg_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)
        self.encoder_k = resnet.resnet50_seg_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)

        self.cls_head = nn.Sequential(nn.Linear(258, 128), nn.ReLU(), nn.Linear(128, 2))
        # self.encoder_q = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_q)
        # self.encoder_k = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_k)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)



        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_pos", torch.randn(dim, K))
        self.register_buffer("queue_neg", torch.randn(dim, K))
        self.queue_pos = nn.functional.normalize(self.queue_pos, dim=0)
        self.queue_neg = nn.functional.normalize(self.queue_neg, dim=0)
        self.register_buffer("queue_pos_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_neg_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pos(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)

        # try:
        batch_size = keys.shape[0]
        ptr = int(self.queue_pos_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_pos[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_pos_ptr[0] = ptr


    @torch.no_grad()
    def _dequeue_and_enqueue_neg(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_neg_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_neg[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_neg_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # import pdb
        # pdb.set_trace()
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        #         x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        #         gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x[idx_unshuffle]

    def forward(self, im_q, im_k, q_label, age, sex, train_cl=False, val=False, bn_shuffle=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        if train_cl is False:
            with torch.no_grad():
                q = self.encoder_q.get_hidden_vec(im_q)
                q = nn.functional.normalize(q, dim=1)
            # import pdb
            # pdb.set_trace()
            multi_q = torch.cat([q, torch.unsqueeze(age, dim=-1), torch.unsqueeze(sex, dim=-1)], dim=-1)
            res = self.cls_head(multi_q)
            return res
        else:
            q, pred_1 = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                if not val:
                    self._momentum_update_key_encoder()  # update the key encoder
                # shuffle for making use of BN
                if bn_shuffle:
                    im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                    k, _ = self.encoder_k(im_k)  # keys: NxC
                    k = nn.functional.normalize(k, dim=1)
                    # undo shuffle
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                else:
                    k, _ = self.encoder_k(im_k)  # keys: NxC
                    k = nn.functional.normalize(k, dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            if q_label == 1:
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue_neg.clone().detach()])
                # dequeue and enqueue
                if not val:
                    self._dequeue_and_enqueue_pos(k)
            else:
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue_pos.clone().detach()])
                # dequeue and enqueue
                if not val:
                    self._dequeue_and_enqueue_neg(k)

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()


            return logits, labels, pred_1


class MoCo_raw(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=128, K=160, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_raw, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet.resnet50_raw_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)
        self.encoder_k = resnet.resnet50_raw_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)
        # self.encoder_q = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_q)
        # self.encoder_k = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_k)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_pos", torch.randn(dim, K))
        self.register_buffer("queue_neg", torch.randn(dim, K))
        self.queue_pos = nn.functional.normalize(self.queue_pos, dim=0)
        self.queue_neg = nn.functional.normalize(self.queue_neg, dim=0)
        self.register_buffer("queue_pos_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_neg_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pos(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_pos_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_pos[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_pos_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_neg(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_neg_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_neg[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_neg_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # import pdb
        # pdb.set_trace()
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        #         x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        #         gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x[idx_unshuffle]

    def forward(self, im_q, im_k, q_label, val=False, train_cl=True, bn_shuffle=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, _ = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # import pdb
        # pdb.set_trace()
        # compute key features
        with torch.no_grad():  # no gradient to keys
            if not val:
                self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            if bn_shuffle:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            else:
                k, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if q_label == 1:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_neg.clone().detach()])
            # dequeue and enqueue
            if not val:
                self._dequeue_and_enqueue_pos(k)
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_pos.clone().detach()])
            # dequeue and enqueue
            if not val:
                self._dequeue_and_enqueue_neg(k)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels


class MoCo_vae(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=128, K=160, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_vae, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet.resnet50_vae_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)
        self.encoder_k = resnet.resnet50_vae_moco(
                            proj_dim=128,
                            num_cls_classes=2,
                            droprate=0)
        # self.encoder_q = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_q)
        # self.encoder_k = model_io.load_retrained_model('./external/pretrain/resnet_50.pth', self.encoder_k)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_pos", torch.randn(dim, K))
        self.register_buffer("queue_neg", torch.randn(dim, K))
        self.queue_pos = nn.functional.normalize(self.queue_pos, dim=0)
        self.queue_neg = nn.functional.normalize(self.queue_neg, dim=0)
        self.register_buffer("queue_pos_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_neg_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pos(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_pos_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_pos[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_pos_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_neg(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_neg_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_neg[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_neg_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # import pdb
        # pdb.set_trace()
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        #         x_gather = concat_all_gather(x)
        batch_size_all = x.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        #         gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x[idx_unshuffle]

    def forward(self, im_q, im_k, q_label, val=False, bn_shuffle=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        [q, _], recon, mu, log_var = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if not val:
                self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            # shuffle for making use of BN
            if bn_shuffle:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                [_, k], _, _, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            else:
                [_, k], _, _, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # import pdb
        # pdb.set_trace()
        if q_label == 1:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_neg.clone().detach()])
            # dequeue and enqueue
            if not val:
                self._dequeue_and_enqueue_pos(k)
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_pos.clone().detach()])
            # dequeue and enqueue
            if not val:
                self._dequeue_and_enqueue_neg(k)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels, recon, mu, log_var


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# if __name__ == '__main__':
    # import torchvision.models as modelso
    #
    # aa = torch.randn((16, 3, 256, 256))
    # bb = torch.randn((16, 3, 256, 256))
    # model = MoCo(models.__dict__["resnet50"])
