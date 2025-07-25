


## Test Codes
## Test Codes
To perform Multi-label Thoracic Disease Classification using Our pre-trained model (Resnet-18 Model).

Test On CheXpert_Train Dataset
```
python test_CheXpert_train.py
```

Test On CheXpert_Train Dataset
```
python test_MIMIC_test.py
```

## Train Codes

### Train Xray Classification Model

```
python main.py --exp_name R50_Single_lstm --model R50_Single_lstm --ClassifyScan_only \
				          --downpatch 256 --crop_sz 256 \
					  --tr_plot train_plot.npy --vl_plot valid_plot.npy \
					  --epochs 25 --batch_size 16 --batch_size_val 64 --lr 1e-4 --decay 6+12+18 --gamma 0.2 \
					  --datapath ../data/MIMIC-JPG
```



