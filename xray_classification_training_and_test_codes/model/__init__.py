import os
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        module = import_module('model.' + args.model)
        if args.Classify_only: self.classifier = module.make_model(args)

    def forward(self, x, itype):
        if itype=='classifier':
            x = self.classifier(x)
        if itype=='encoder':
            x = self.encoder(x)
        if itype=='decoder':
            x = self.decoder(x)
        if itype=='decoder_inf':
            x = self.decoder.sample_train(x)
        return x
