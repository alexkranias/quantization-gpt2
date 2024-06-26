# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'instantnet'


"""Data Dir and Weight Dir"""
C.dataset_path = "/home/yf22/dataset/" # root dir of cifar100

C.dataset = 'cifar100'

if C.dataset == 'cifar10':
    C.num_classes = 10
elif C.dataset == 'cifar100':
    C.num_classes = 100
else:
    print('Wrong dataset.')
    sys.exit()

"""Image Config"""

C.num_train_imgs = 50000
C.num_eval_imgs = 10000

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""


C.opt = 'Sgd'

C.momentum = 0.9
C.weight_decay = 5e-4

C.betas=(0.5, 0.999)
C.num_workers = 8


""" Search Config """
C.grad_clip = 5

C.pretrain = False
# C.pretrain = 'ckpt/finetune/weights_199.pt'

# C.bit_schedule = 'high2low'
# C.bit_schedule = 'low2high'
# C.bit_schedule = 'sandwich'
C.bit_schedule = 'avg_loss'
# C.bit_schedule = 'max_loss'

C.dws_chwise_quant = True

# C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
C.stride_list = [1, 1, 2, 2, 1, 2, 1]

C.stem_channel = 16
C.header_channel = 1504

C.num_bits_list = [4, 8, 12, 16, 32]

# C.loss_scale = [8, 4, 8/3, 2, 1]
C.loss_scale = [1, 1, 1, 1, 1]

C.distill_weight = 1

C.cascad = True

C.update_bn_freq = None

C.num_bits_list_schedule = None
# C.choice_list_schedule = 'high2low'
# C.choice_list_schedule = 'low2high'

C.schedule_freq = 20

########################################

C.batch_size = 128  #96
C.niters_per_epoch = C.num_train_imgs // C.batch_size
C.image_height = 32 # this size is after down_sampling
C.image_width = 32

C.save = "finetune"
########################################

if C.pretrain == True:
    C.num_bits_list = [0]

    C.nepochs = 200  # 600

    C.eval_epoch = 1

    # C.lr_schedule = 'multistep'
    # C.lr = 1e-1

    C.lr_schedule = 'cosine'
    C.lr = 0.025  # 0.01

    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [80, 120, 160]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    C.load_path = 'ckpt/search'

    C.eval_only = False

else:
    C.nepochs = 200

    C.eval_epoch = 1

    # C.lr_schedule = 'multistep'
    # C.lr = 1e-1

    C.lr_schedule = 'cosine'
    C.lr = 0.025

    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [80, 120, 160]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    C.load_path = 'ckpt/search'

    C.eval_only = False
    C.update_bn = True
    C.show_distrib = True

    C.finetune_bn = False
    C.ft_bn_epoch = 10
    C.ft_bn_lr = 1e-3
    C.ft_bn_momentum = 0.9

