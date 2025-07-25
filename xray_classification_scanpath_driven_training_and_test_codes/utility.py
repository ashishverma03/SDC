import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


class checkpoint():
    def __init__(self, args):

        self.args = args
        self.dir = os.path.join('../MIT_Save_SingleScan_FinalCVPR/', args.exp_name)
        if args.reset:
            os.system('rm -rf ' + self.dir)

    
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join('../MIT_Save_SingleScan_FinalCVPR/', args.exp_name), exist_ok=True)



        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(args.model + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def write_log(self, log, refresh=False):
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()






