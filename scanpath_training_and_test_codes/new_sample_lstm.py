import torch
import numpy as np 
import math
import argparse
import pickle 
import os
from torchvision import transforms 
import torchvision
from new_test_data_loader import get_loader 
from lstm_model import EncoderCNN, DecoderRNN
from PIL import Image
import matplotlib.pyplot as plt
import multimatch_gaze as m
import time
import pdb


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  
                             

    test_data_loader = get_loader(args.image_resize, args.feat_size, args.test_path, transform)


    classID = np.zeros([args.image_resize,args.image_resize])
    div_factor = args.image_resize/args.feat_size
    for i in range(1,args.image_resize):
        for j in range(1,args.image_resize):
            classID[i,j] =  math.ceil(i/16)+(math.ceil(j/16)-1)*div_factor
    

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  
    decoder = DecoderRNN(args.embed_size, args.hidden_size, args.feat_size, args.num_layers).eval() 
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    curr_epoch=200

    mm_score_epoch = np.zeros((1,4))

    for epoch in range(curr_epoch,args.num_epochs):

        snapshot_name = 'epoch_%d' % (epoch)
        encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder' + snapshot_name + '.ckpt')))
        decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'decoder' + snapshot_name + '.ckpt')))


        total_step = len(test_data_loader)
        class_seqID = np.zeros((total_step,200))

        # Prepare an image
        for i, (images, image_basename) in enumerate(test_data_loader):

            image_tensor = images.to(device)
            
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          
            class_seqID[i,:] = sampled_ids
            last_pts = np.where(sampled_ids == args.image_resize+1)
            if np.any(last_pts):
                last_ptID = last_pts[0][0]
                
            else:
                last_ptID = 199

            traj_classID =  sampled_ids[1:last_ptID]
            traj = np.zeros((3,len(traj_classID)))

            for ind in range(len(traj_classID)):
                class_id = traj_classID[ind]
                locations = np.where(classID == class_id)
                traj[0, ind] = locations[0][127]
                traj[1, ind] = locations[1][7]
 
            np.save('result_scanpaths/'+ image_basename[0][:-4],traj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--image_resize', type=int, default=256 , help='input image to be resized to')
    parser.add_argument('--test_path', type=str,default = 'dataset/new_test/', help='input image for generating caption')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--num_epochs', type=int, default=201)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--feat_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
