
import os
import pdb

os.system('python main.py --exp_name R50_Single_lstm --model R50_Single_lstm --ClassifyScan_only \
						  --downpatch 256 --crop_sz 256 \
						  --tr_plot train_plot.npy --vl_plot valid_plot.npy \
						  --epochs 25 --batch_size 16 --batch_size_val 64 --lr 1e-4 --decay 6+12+18 --gamma 0.2 \
						  --datapath /home1/ChestXray/data/MIMIC-JPG ')


"""
os.system('python main.py --exp_name D201_SingleScanImgFeat_Transformer --model D201_SingleScanImgFeat --ClassifyScan_only \
						  --downpatch 256 --crop_sz 256 \
						  --tr_plot train_plot.npy --vl_plot valid_plot.npy \
						  --epochs 25 --batch_size 16 --batch_size_val 64 --lr 1e-4 --decay 6+12+18 --gamma 0.2 \
						  --datapath /home1/ChestXray/data/MIMIC-JPG ')
"""
"""
os.system('python test.py --datapath /home1/ChestXray/data/NIH_data --split_id 1 --exp_name R18F8 \
						  --scanimage_resize 256 --downpatch 256 --crop_sz 0 \
						  --scanfeat_sz 1024 --max_seq_length 80 --hidden_size 512 --num_layers 1 --feat_size 8 \
						  --model ScanR18 --kernel 30 --sigma 15 --embed_size 256 ')

"""




"""
**** only Resnet without scanpaths Best Loss: 0.834005

R50_emb_lstm_featsz16 : Resnet 50, linear Embedding used, lstm, patchsize is 16x16 (model = R50_Scan1.py)

R50_emb_transformer_featsz16 : Resnet 50, linear Embedding (dim=2048) used, Transformer, patchsize is 16x16 (model = R50_ScanT1.py)

R50_emb_transformer_featsz16_v1: Resnet 50, linear Embedding (dim=256) used, Transformer, patchsize is 16x16 (model = R50_ScanT1_v1.py)

R50_feat_lstm_featsz16_2 : Resnet 50, resnet features for representing patches, lstm (1 layer), patchsize is 16x16 (model = R50_Scan2.py)

R50_feat_lstm2_featsz16_2 : Resnet 50, resnet features for representing patches, lstm (2 layer), patchsize is 16x16 (model = R50_Scan2.py)

R50_feat_transformer_featsz16_2 : Resnet 50, resnet features for representing patches, Transformer, patchsize is 16x16 (model = R50_ScanT2.py)

R50_feat_transformer_featsz16_2_v1: Resnet 50, resnet features for representing patches, Transformer(depth=8, mlp_dim=64), patchsize is 16x16 (model = R50_ScanT2_v1.py)
"""
