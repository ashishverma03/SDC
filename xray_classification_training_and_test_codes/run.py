
import os
import pdb



os.system('python main.py --exp_name NoAug_Pad_R152 --model R152 --Classify_only \
						  --downpatch 256 --crop_sz 256 \
						  --tr_plot train_plot.npy --vl_plot valid_plot.npy \
						  --epochs 25 --batch_size 16 --batch_size_val 64 --lr 1e-4 --decay 6+12+18 --gamma 0.2 \
						  --datapath data/MIMIC-JPG ')


