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


def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).
	
	We should build custom collate_fn rather than using default collate_fn, 
	because merging caption (including padding) is not supported in default.

	Args:
		data: list of tuple (image, caption). 
			- image: torch tensor of shape (3, 256, 256).
			- caption: torch tensor of shape (?); variable length.

	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: len(x[3]), reverse=True)

	img, imgLabel, imgLabel_uncer, scanpath = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	img = torch.stack(img, 0)
	imgLabel = torch.stack(imgLabel, 0)
	imgLabel_uncer = torch.stack(imgLabel_uncer, 0)

	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(cap) for cap in scanpath]
	targets = torch.zeros(len(scanpath), max(lengths)).long()
	for i, cap in enumerate(scanpath):
		end = lengths[i]
		targets[i, :end] = cap[:end]        
	return img, imgLabel, imgLabel_uncer, targets, lengths

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
		

		if args.Classify_only or args.do_all or args.ClassifyScan_only:
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

	def FixationtoclassID(self, traj, org_img_sz, feat_sz):

		traj = traj[:,traj.min(axis=0)>=0]

		div_factor = org_img_sz/feat_sz
		regionID_seq = np.zeros(len(traj[0]))
		p = 0
		for pt in range(len(traj[0])):
			
			px = traj[0,pt]
			py = traj[1,pt]
			if 0<px<256 and 0<py<256:
			 
				m = np.ceil(px/feat_sz)
				n = np.ceil(py/feat_sz)
				region_id = m+(n-1)*div_factor
				regionID_seq[p] = region_id 
				p = p+1
		regionID_seq = regionID_seq[regionID_seq<256]
		#print(np.max(regionID_seq))
		return regionID_seq

	def __getitem__(self, index):
		istr = self.ifiles[index]
		istr = istr.split("|")

		ipath = self.path_bin+'/'+istr[0][:-4]+'.pt'

		path_captions = '/home1/ChestXray/LSTM/LSTM_REFLACX/MIMIC-JPG/'

		if self.is_train:

			scanpath = np.load(path_captions + 'train_pad_scan/' + istr[0][:-4] + '.npy')
		else:

			scanpath = np.load(path_captions + 'valid_pad_scan/' + istr[0][:-4] + '.npy')

		scanpath = self.FixationtoclassID(scanpath, 256, 16)
		scanpath = np.squeeze(scanpath.astype(int))
		scanpath = torch.from_numpy(scanpath)



		with open(ipath, 'rb') as _f:
			img = pickle.load(_f)
		img = self._np2Tensor(img)
		'''
		if self.is_train:
			img = self._randomCrop(img)
			img = self.horizontal_flip(img)
			#img = self.rotate(img)
		else:
			img = self._centerCrop(img)
		'''
		imgLabel = self._tolabel(istr[1])
		imgLabel_uncer = torch.where(imgLabel<0, 0, 1)

		if self.args.Classify_only or self.args.ClassifyScan_only:
			return img, imgLabel, imgLabel_uncer, scanpath


	def __len__(self):
		return len(self.ifiles)



class Data:
	def __init__(self, args):
		val_dataset = CustomDataset(args, is_train=False)
		self.loader_valid = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.n_threads, drop_last=False, collate_fn=collate_fn)
		if not args.valid_only:
			if args.Classify_only or args.ClassifyScan_only:
				train_dataset = CustomDataset(args, is_train=True)
				self.loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads, drop_last=True, collate_fn=collate_fn)
		

