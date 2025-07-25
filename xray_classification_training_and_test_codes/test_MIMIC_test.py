import torch

import os
import sys
from math import exp
import torch
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
from PIL import Image
import math
import random
import numpy as np
import time
from decimal import Decimal
from sklearn.metrics.ranking import roc_auc_score

import model
from option import args

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import cv2
import pdb


def computeAUROC (GT, PRED, classCount):
	outAUROC = []
		
	for i in range(classCount):
		outAUROC.append(roc_auc_score(GT[:, i], PRED[:, i], average = 'weighted'))
			
	return outAUROC


def UNremoved_computeAUROC(GT, PRED, classCount, UN):
	outAUROC, _GT, _PRED = [], [], []
		
	for i in range(classCount):
		for j in range(GT.shape[0]):
			if UN[j, i]!=0:
				_GT.append(GT[j, i])
				_PRED.append(PRED[j, i])
		outAUROC.append(roc_auc_score(np.array(_GT), np.array(_PRED), average = 'weighted'))
			
	return outAUROC

def _tolabel(_label):
	_label = _label.split(",")
	_label = list(map(int, _label)) 
	_label = np.array(_label)
	_label = torch.from_numpy(_label).long()
	return _label

def _np2Tensor(img):
	img = np.stack((img, img, img), 2)
	img = np.ascontiguousarray(img.transpose((2, 0, 1)))
	img = torch.from_numpy(img).float()
	return torch.unsqueeze(img, 0)


args.datapath = '../data/MIMIC-JPG'
args.Classify_only = True
savepath = './trained_models/'
args.model = 'R18'
test_txt_file = args.datapath+'/images/test.txt'
test_image_path = args.datapath+'/images/test_pad_pt/'
result_savepath = 'MIMIC_test.txt'
model_dir = savepath+args.model+'_best.pth.tar'

_model = model.Model(args)
checkpoint = torch.load(model_dir)
_model.load_state_dict(checkpoint['model'])
_model = _model.cuda()
_model.eval()


ifiles=[]
f = open(test_txt_file, "r")
for line in f.readlines():
	line = line[:-1]
	ifiles.append(line)


GT = []
PRED = []
UN = []
for index in range(len(ifiles)): 
	sys.stdout.write("Calculation in Progress: %d/ %d  \r" % (index+1, len(ifiles)))
	sys.stdout.flush()

	istr = ifiles[index]
	istr = istr.split("|")

	ipath = test_image_path+istr[0][:-4]+'.pt'
	with open(ipath, 'rb') as _f:
		_img = pickle.load(_f)
	_img = _np2Tensor(np.array(_img))
	_img = _img/255.0
	_img = _img.cuda()


	IMlabel = _tolabel(istr[1]).unsqueeze_(0)
	IMlabel_uncer = torch.where(IMlabel<0, 0, 1)

	with torch.no_grad():
		starttime = time.time()
		out = _model(_img, 'classifier')
		print(time.time() - starttime)


	GT.extend(IMlabel.numpy())
	PRED.extend(out.data.cpu().numpy())
	UN.extend(IMlabel_uncer.numpy())

GT, PRED, UN = np.stack(GT, 0), np.stack(PRED, 0), np.stack(UN, 0)

outROC = UNremoved_computeAUROC(GT, PRED, 14, UN)


all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', \
			               'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']



np.save(savepath+'GT_MIMIC_test.npy', GT)
np.save(savepath+'PRED_MIMIC_test.npy', PRED)
np.save(savepath+'UN_MIMIC_test.npy', UN)




f = open(savepath+result_savepath, 'w')

f.write('\n\n'+args.model + '\n')
f.write('----------------------------------------\n')


print(outROC)
avg_auc = 0
for i in range(len(outROC)):
	print(all_labels[i], outROC[i])
	f.write('%s:   %f'%(all_labels[i], outROC[i]))
	f.write('\n')
	avg_auc = avg_auc + outROC[i] 


f.write('\n\n')

print(avg_auc/len(outROC))

f.write('Average AUCROC:  %f'%(avg_auc/len(outROC)))

f.close()