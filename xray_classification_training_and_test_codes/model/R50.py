import torch
import torch.nn as nn
import torchvision


def make_model(args, parent=False):
	return main_network()

class MeanShift(nn.Conv2d):
	def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False

class main_network(nn.Module):
	def __init__(self):
		super(main_network, self).__init__()

		rgb_mean = (0.485, 0.456, 0.406)
		rgb_std = (0.229, 0.224, 0.225)
		
		self.main_model = torchvision.models.resnet50(pretrained=True)

		kernelCount = self.main_model.fc.in_features
		self.main_model.fc = nn.Sequential(nn.Linear(kernelCount, 14), nn.Sigmoid())

		self.sub_mean = MeanShift(rgb_mean, rgb_std)

	def forward(self, x):
		x = self.sub_mean(x)
		x = self.main_model(x)
		return x