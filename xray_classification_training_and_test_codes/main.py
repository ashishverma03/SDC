import torch
import model
from option import args
from trainer import Trainer
from dataloader import Data
import numpy as np
import random
import time
import utility
import pdb

#torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = True

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
_checkpoint = utility.checkpoint(args)

def main():
	global model
	loader = Data(args)
	_model = model.Model(args)
	_optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
	t = Trainer(args, loader, _model, _optimizer, _checkpoint)
	if args.valid_only:
		t.test()
	else:
		while not t.terminate():
			if args.Classify_only:
				t.train_Classifier()
				#t.test_Classifier()
			if args.Scan_only:
				t.train_Scan()
				t.test_Scan()
			if args.Saliency_only:
				t.train_Saliency()
				t.test_Saliency()

if __name__ == '__main__':
	main()
