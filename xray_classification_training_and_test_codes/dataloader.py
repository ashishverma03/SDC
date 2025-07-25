import os
import sys
import glob
import torch
import torchvision
import time
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
import imageio
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import pdb
from torchvision.utils import save_image
import copy


class CustomDataset(Dataset):
	def __init__(self, args, is_train):


		self.is_train = is_train
		self.args = args
		self.rotate = torchvision.transforms.RandomRotation(degrees=(-15, +15))
		self.jitter = torchvision.transforms.ColorJitter(brightness=0, contrast=0.25, saturation=0, hue=0)
		self.resize = torchvision.transforms.Resize(size=(args.scanimage_resize, args.scanimage_resize), interpolation=Image.BICUBIC, antialias=True)
		self.horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
		self.vertical_flip = torchvision.transforms.RandomVerticalFlip(p=0.5)
		self.all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', \
			               'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
		

		if args.Classify_only or args.do_all:
			self.ifiles=[]
			if self.is_train:
				_ = self._save('train.txt', 'train_pad')
				f = open(os.path.join(self.args.datapath, 'images', 'train.txt'), "r")
			else:
				_ = self._save('valid.txt', 'valid_pad')
				f = open(os.path.join(self.args.datapath, 'images', 'valid.txt'), "r")

			for line in f.readlines():
				line = line[:-1]
				self.ifiles.append(line)


	def _save(self, ifilename, itype):
		self.path_bin = os.path.join(self.args.datapath, 'bin', itype, 'X'+str(self.args.downpatch))
		os.makedirs(self.path_bin, exist_ok=True)
		trainval_files = []
		f = open(os.path.join(self.args.datapath, 'images', ifilename), "r")
		for line in f.readlines():
			line = line[:-1]
			line = line.split("|")
			trainval_files.append(line[0])

		for i in tqdm(range(len(trainval_files)), desc="Making file..."):
			if os.path.exists(self.path_bin+'/'+trainval_files[i][:-4]+'.pt')==False:
				_img = Image.open(self.args.datapath+'/images/'+itype+'/'+trainval_files[i]).convert('L')
				_img = _img.resize((self.args.downpatch, self.args.downpatch), Image.BICUBIC)
				_img = np.uint8(np.array(_img))
				with open(self.path_bin+'/'+trainval_files[i][:-4]+'.pt', 'wb') as _f:
					pickle.dump(_img, _f)

		return 1



	def _randomCrop(self, img):
		x1 = random.randint(0, self.args.downpatch - self.args.crop_sz)
		y1 = random.randint(0, self.args.downpatch - self.args.crop_sz)
		img = img[:, x1:x1+self.args.crop_sz, y1:y1+self.args.crop_sz]
		return img

	def _centerCrop(self, img):
		x1 = int(img.shape[1]/2)
		y1 = int(img.shape[2]/2)
		pz = int(self.args.crop_sz/2)
		img = img[:, x1-pz:x1+pz, y1-pz:y1+pz]
		return img

	def _tolabel(self, _label):
		_label = _label.split(",")
		_label = list(map(int, _label)) 
		_label = np.array(_label)
		_label = torch.from_numpy(_label).long()
		return _label

	def _np2Tensor(self, img):
		img = np.stack((img, img, img), 2)
		img = np.ascontiguousarray(img.transpose((2, 0, 1)))
		img = torch.from_numpy(img).float()
		return img

	def __getitem__(self, index):
		istr = self.ifiles[index]
		istr = istr.split("|")

		ipath = self.path_bin+'/'+istr[0][:-4]+'.pt'

		with open(ipath, 'rb') as _f:
			img = pickle.load(_f)
		img = self._np2Tensor(img)

		imgLabel = self._tolabel(istr[1])
		imgLabel_uncer = torch.where(imgLabel<0, 0, 1)

		if self.args.Classify_only:
			return img, imgLabel, imgLabel_uncer


	def __len__(self):
		return len(self.ifiles)



class Data:
	def __init__(self, args):
		val_dataset = CustomDataset(args, is_train=False)
		self.loader_valid = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.n_threads, drop_last=False)
		if not args.valid_only:
			if args.Classify_only:
				train_dataset = CustomDataset(args, is_train=True)
				self.loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads, drop_last=True)
		

