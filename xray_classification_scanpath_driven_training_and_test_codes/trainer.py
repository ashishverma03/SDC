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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from scipy.ndimage import gaussian_filter
from math import exp
import torch.nn.functional as F
import pdb


class Trainer():
	def __init__(self, args, my_loader, my_model, my_optimizer, my_checkpoint):
		self.args = args
		if not args.valid_only:
			self.loader_train = my_loader.loader_train
		self.loader_valid = my_loader.loader_valid
		self.model = my_model
		self.model = self.model.cuda()
		self.ckp = my_checkpoint
		self.savedir = os.path.join('../MIT_Save_SingleScan_FinalCVPR/', self.args.exp_name)
		self.tr_plot, self.vl_plot = [], []
		self.optimizer = my_optimizer
		self.bestvalloss = 0
		self.bestepoch = 0
		self.scanloss = nn.CrossEntropyLoss()
		self.decay = args.decay.split('+')
		self.decay = [int(i) for i in self.decay]
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay, gamma=args.gamma)
		if args.resume or args.valid_only:
			checkpoint = torch.load(args.trained_model)
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.scheduler.load_state_dict(checkpoint['scheduler'])
			self.bestvalloss = checkpoint['loss']
			self.bestepoch = checkpoint['epoch']
			if os.path.exists(self.savedir+'/'+self.args.tr_plot):
				self.tr_plot = np.load(self.savedir+'/'+self.args.tr_plot).tolist()
				self.vl_plot = np.load(self.savedir+'/'+self.args.vl_plot).tolist()
			print(self.scheduler.last_epoch)


		if args.pre_train:
			checkpoint = torch.load(args.trained_model)
			self.model.load_state_dict(checkpoint['model'])

		self.class_weights = np.zeros(14)
		total_samples = 0
		self.uncertain_class_weights = np.zeros(14)
		f = open(os.path.join(self.args.datapath, 'images', 'train.txt'), "r")
		total_samples = 0
		for line in f.readlines():
			line = line[:-1]
			line = line.split("|")
			line = line[1]
			line = line.split(",")
			for j in range(len(line)):
				total_samples += 1
				if int(line[j])==1:
					self.class_weights[j] += 1
				if int(line[j])==-1:
					self.uncertain_class_weights[j] += 1

		self.class_weights = 1 - np.divide(self.class_weights, np.ones(14)*total_samples - self.uncertain_class_weights)
		self.class_weights = torch.from_numpy(self.class_weights)
		self.class_weights = torch.unsqueeze(self.class_weights, 0).cuda().float()


	def UnRemoved_WeightedCrossEntropy(self, yhat, y, c_weights, y_un):
		
		batch_size = yhat.size()[0]
		n_classes = yhat.size()[1]		

		pos_wt = torch.cat([c_weights]*batch_size, 0)
		
		neg_wt = 1 - pos_wt
		wt = (y*pos_wt + (1 - y)*(neg_wt))*y_un
		
		unweighted_loss = - y*(torch.log(yhat + 1e-10)) - (1 - y)*(torch.log(1 - yhat + 1e-10))
		weighted_loss = unweighted_loss*wt

		loss = weighted_loss.sum()

		return loss

	def WeightedCrossEntropy(self, yhat, y, c_weights):
		
		batch_size = yhat.size()[0]
		n_classes = yhat.size()[1]		

		pos_wt = torch.cat([c_weights]*batch_size, 0)
		
		neg_wt = 1 - pos_wt
		wt = y*pos_wt + (1 - y)*(neg_wt)
		
		unweighted_loss = - y*(torch.log(yhat + 1e-10)) - (1 - y)*(torch.log(1 - yhat + 1e-10))
		weighted_loss = unweighted_loss*wt

		loss = weighted_loss.sum()

		return loss


	def computeAUROC(self, GT, PRED, classCount):
		outAUROC = []
			
		for i in range(classCount):
			outAUROC.append(roc_auc_score(GT[:, i], PRED[:, i], average = 'weighted'))
				
		return outAUROC

	def UNremoved_computeAUROC(self, GT, PRED,classCount, UN):
		outAUROC, _GT, _PRED = [], [], []
			
		for i in range(classCount):
			for j in range(GT.shape[0]):
				if UN[j, i]!=0:
					_GT.append(GT[j, i])
					_PRED.append(PRED[j, i])
			outAUROC.append(roc_auc_score(np.array(_GT), np.array(_PRED), average = 'weighted'))
				
		return outAUROC

	def train_Classifier(self):
		self.model.train()
		train_xray_loss = 0
		lr = self.optimizer.param_groups[0]['lr']
		print('===========================================\n')
		print('[Epoch: %d] [Learning Rate: %.2e]'%(self.scheduler.last_epoch+1, Decimal(lr)))
		self.ckp.write_log('===========================================\n')
		self.ckp.write_log('[Epoch: %d] [Learning Rate: %.2e]'%(self.scheduler.last_epoch+1, Decimal(lr)))
		startepoch = time.time()
		for i_batch, (xrayimages, xraylabels, xraylabels_uncer, xrayscan, xraylengths) in enumerate(self.loader_train):


			xrayimages, xraylabels, xraylabels_uncer, xrayscan = xrayimages.cuda(), xraylabels.cuda(), xraylabels_uncer.cuda(), xrayscan.cuda()
			self.optimizer.zero_grad()
			out = self.model((xrayimages/255.0, xrayscan, xraylengths), 'classifierScan')

			loss = self.UnRemoved_WeightedCrossEntropy(out, xraylabels, self.class_weights, xraylabels_uncer)
			loss.backward()
			self.optimizer.step()



			train_xray_loss += loss.data.cpu()
			self.tr_plot.append(loss.data.cpu())

			if (i_batch+1)%100==0:
				train_xray_loss = train_xray_loss/100
				total_time = time.time()-startepoch
				startepoch = time.time()
				print('[Batch: %d / %d] [Classify Loss: %f] [Time: %.1f s]'%(i_batch+1, len(self.loader_train), train_xray_loss, total_time))
				self.ckp.write_log('[Batch: %d / %d] [Classify Loss: %f] [Time: %.1f s]'%(i_batch+1, len(self.loader_train), train_xray_loss, total_time))
				train_xray_loss = 0  

			if i_batch%2000==0: 
				self.test_Classifier()
				self.model.train()
				torch.set_grad_enabled(True)

		np.save(self.savedir+'/'+self.args.tr_plot, np.array(self.tr_plot))
		self.scheduler.step()

	def test_Classifier(self):
		torch.set_grad_enabled(False)
		self.model.eval()
		val_loss = 0
		GT, PRED, UN = [], [], []
		for i_batch, (xrayimages, IMlabel, IMlabel_uncer, xrayscan, xraylengths) in enumerate(self.loader_valid):

			sys.stdout.write("Calculation in Progress: %d/ %d  \r" % (i_batch+1, len(self.loader_valid)))
			sys.stdout.flush()

			xrayimages, IMlabel, IMlabel_uncer, xrayscan = xrayimages.cuda(), IMlabel.cuda(), IMlabel_uncer.cuda(), xrayscan.cuda()
			torch.cuda.empty_cache()
			with torch.no_grad():
				out = self.model((xrayimages/255.0, xrayscan, xraylengths), 'classifierScan')

			GT.extend(IMlabel.cpu().numpy())
			PRED.extend(torch.squeeze(out.data.cpu()).numpy())
			UN.extend(IMlabel_uncer.cpu().numpy())

		GT, PRED, UN = np.stack(GT, 0), np.stack(PRED, 0), np.stack(UN, 0)

		outROC = self.UNremoved_computeAUROC(GT, PRED, 14, UN)
		out = 0
		for ix in range(len(outROC)):
			out = out + outROC[ix]
		val_loss = out/len(outROC)

		self.vl_plot.append(val_loss)
		torch.set_grad_enabled(True)
		#self.scheduler.step(val_loss)
		if not self.args.valid_only:
			if self.bestvalloss<val_loss:
				self.bestepoch = self.scheduler.last_epoch
				self.bestvalloss = val_loss
				torch.save({'model': self.model.state_dict(), 'loss': self.bestvalloss, 'epoch': self.bestepoch,
							'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
							}, self.savedir+'/'+self.args.model+'_best.pth.tar')
			torch.save({'model': self.model.state_dict(), 'loss': self.bestvalloss, 'epoch': self.bestepoch,
						 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
						}, self.savedir+'/'+self.args.model+'_check.pth.tar')

		print('\n\nValid Loss:    %f     (Best Loss: %f at %d)'%(val_loss, self.bestvalloss, self.bestepoch))
		self.ckp.write_log('\n\nValid Loss:    %f     (Best Loss: %f at %d)'%(val_loss, self.bestvalloss, self.bestepoch), refresh=True)
		np.save(self.savedir+'/'+self.args.vl_plot, np.array(self.vl_plot))


	def terminate(self):
		if self.args.valid_only:
			self.test()
			return True
		else:
			epoch = self.scheduler.last_epoch
			return epoch >= self.args.epochs


