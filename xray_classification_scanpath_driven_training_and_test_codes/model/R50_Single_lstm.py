import torch
import torch.nn as nn
import torchvision
from .attention import Attention
import pdb

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


def make_model(args, parent=False):
	return main_network_scan()

class MeanShift(nn.Conv2d):
	def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False

class main_network_scan(nn.Module):
	def __init__(self):
		super(main_network_scan, self).__init__()

		rgb_mean = (0.485, 0.456, 0.406)
		rgb_std = (0.229, 0.224, 0.225)
		
		model = torchvision.models.resnet50(pretrained=True)
		self.kernelCount = model.fc.in_features

		rnn_hidden_dim = self.kernelCount
		num_directions = 1

		self.main_model = torch.nn.Sequential(*(list(model.children())[:-2]))
		self.pool = nn.AdaptiveAvgPool2d(1)
	
		self.lstm = nn.LSTM(self.kernelCount, hidden_size=rnn_hidden_dim, num_layers=1, batch_first=True)
		self.attention = Attention(query_dim=rnn_hidden_dim // num_directions)

		self.classifier = nn.Sequential(nn.Linear(2*self.kernelCount, 14), nn.Sigmoid())

		self.sub_mean = MeanShift(rgb_mean, rgb_std)
		self.upscale = torch.nn.Upsample(scale_factor=2, mode='bilinear')

	def forward(self, x):
		imgs, scanpaths, lengths = x

		feat = self.sub_mean(imgs)
		feat = self.main_model(feat)
		up_feat = self.upscale(feat)
		
		imgfeat = up_feat.permute(0,2,3,1)
		imgfeat = imgfeat.flatten(1,2)

		imgfeat_shape = imgfeat.shape
		# pdb.set_trace()
		zer_pad = torch.zeros((imgfeat_shape[0], 1, imgfeat_shape[2]))
		zer_pad = zer_pad.cuda()
		imgfeat = torch.cat([zer_pad, imgfeat], dim=1)

		positions  = scanpaths.unsqueeze(2).expand(scanpaths.size(0), scanpaths.size(1), imgfeat.size(2))
		imgfeat_seq = torch.gather(imgfeat, 1, positions)

		packed_imgfeat_seq = torch.nn.utils.rnn.pack_padded_sequence(imgfeat_seq, lengths, batch_first=True)
		packed_output, (ht, ct) = self.lstm(packed_imgfeat_seq)
		hidden = ht[-1]
		output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
		scanfeat = self.attention(hidden, output, output)

		imagefeat= self.pool(feat).view(-1, self.kernelCount)
	
		combfeat = torch.cat([imagefeat, scanfeat], 1)
		out = self.classifier(combfeat)
		
		return out