import argparse

parser = argparse.ArgumentParser(description='Training ChestX Algorithms')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

# Xray data specifications
parser.add_argument('--datapath', default='data/MIMIC-JPG', type=str, help='NIH xray images data')
parser.add_argument('--downpatch', type=int, help='Downscaled xray Image Size for both NIH and MIT') # crop_sz
parser.add_argument('--crop_sz', type=int, help='Cropped xray Image Size for NIH')
parser.add_argument('--exp_name', type=str, help='Experiment Name')

# Scanimages data specifications
parser.add_argument('--scandata_path', default='MIT_data', type=str, help='MIT xray images with scanpath data')
parser.add_argument('--scanimage_resize', type=int, help='Downscaled xray images of MIT for scanmodel training')

# Classifier model specifications
parser.add_argument('--model', help='model name')
parser.add_argument('--trained_model', type=str, help='pre-trained model directory')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--pre_train', action='store_true', help='resume training')
parser.add_argument('--reset', action='store_true', help='Delete all model and plots to start training')

# Scanpath model specifications
parser.add_argument('--scanfeat_sz', type=int, help='Feature size of MIT images for scanmodel training')
parser.add_argument('--max_seq_length', type=int, help='Maximum Sequence length during generation')
parser.add_argument('--embed_size', type=int, help='Embedding Size')

parser.add_argument('--hidden_size', type=int, help='Dimension of hidden state of LSTM')
parser.add_argument('--num_layers', type=int, help='Number of LSTM layers')
parser.add_argument('--feat_size', type=int, help='output dimension of scanpath model')

# Training specifications
parser.add_argument('--epochs', type=int, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, help='input batch size for training')
parser.add_argument('--batch_size_val', type=int, help='input batch size for validation/testing')

parser.add_argument('--valid_only', action='store_true', help='set this option to test the model')
parser.add_argument('--Classify_only', action='store_true', help='')
parser.add_argument('--Scan_only', action='store_true', help='')
parser.add_argument('--Saliency_only', action='store_true', help='')
parser.add_argument('--do_all', action='store_true', help='')

parser.add_argument('--tr_plot', help='train plot name')
parser.add_argument('--vl_plot', help='valid plot name')

# Optimization specifications
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--patience', type=int, help='learning rate decay type')
parser.add_argument('--decay', type=str, help='learning rate decay type')
parser.add_argument('--gamma', type=float, help='learning rate decay factor for step decay')

args = parser.parse_args()

#if args.resume:
#	args.trained_model = '../MIT_Save/'+args.exp_name+'/'+args.model+'_ID_'+str(args.split_id)+'_checkpoint.pth.tar'

if args.valid_only:
	args.trained_model = '../MIT_Save/'+args.exp_name+'/'+args.model+'_ID_'+str(args.split_id)+'_best.pth.tar'


