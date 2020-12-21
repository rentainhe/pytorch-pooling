from types import MethodType
from datetime import datetime
import os
import torch
import numpy as np
import random

class Configs():
    def __init__(self):
        self.epoch = 200
        self.milestones = [60, 120, 160]
        self.save_epoch = 20
        self.gpu='0'
        self.batch_size=64
        self.mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.num_workers = 8
        self.pin_memory = True
        self.eval_every_epoch = True
        self.time = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
        self.seed = random.randint(0, 9999999)
        self.version = str(self.seed)
        self.batch_size = 64
        self.gradient_accumulation_steps = 1
        self.ckpts_dir = 'ckpts'
        self.result_log_dir = 'log'
        self.tensorboard_log_dir = 'runs'
        self.training_init()
        self.path_init()

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def training_init(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.n_gpu = len(self.gpu.split(','))
        self.devices = [_ for _ in range(self.n_gpu)]
        torch.set_num_threads(2)

        # fix seed
        torch.manual_seed(self.seed)
        if self.n_gpu < 2:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Gradient accumulate setup
        assert self.batch_size % self.gradient_accumulation_steps == 0
        self.sub_batch_size = int(self.batch_size / self.gradient_accumulation_steps)
        self.eval_batch_size = int(self.sub_batch_size / 2)

    def path_init(self):
        for attr in dir(self):
            if 'dir' in attr and not attr.startswith('__'):
                if getattr(self,attr) not in os.listdir('./'):
                    os.makedirs(getattr(self, attr))


    def __str__(self):
        # print Hyper Parameters
        settings_str = ''
        for attr in dir(self):
            if not 'np' in attr and not 'random' in attr and not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                settings_str += '{ %-17s }->' % attr + str(getattr(self, attr)) + '\n'
        return settings_str

configs = Configs()




