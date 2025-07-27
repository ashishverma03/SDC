import argparse
import torch
import torch.nn as nn
import numpy as np
import math
import os
import pickle
from lstm_data_loader import get_loader
from testimg_loader import get_loader as val_get_loader 
from lstm_model import EncoderCNN, DecoderRNN, AverageMeter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision import transforms
from torch.autograd import Variable
import pdb


# Device configuration
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu') #'cuda' if torch.cuda.is_available() else

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    

    # Build data loader
    train_data_loader = get_loader(args.image_resize, args.feat_size, args.train_path,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    val_data_loader = val_get_loader(args.val_path,
                             transform, shuffle=False) 

    # Build the models

    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, args.feat_size, args.num_layers).to(device)

    # encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))
    curr_epoch = 0
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    
    for epoch in range(curr_epoch,args.num_epochs):
        train(train_data_loader, encoder, decoder, criterion, optimizer, epoch, args)
        validate(val_data_loader, encoder, decoder, optimizer, epoch, args)


def train(train_data_loader, encoder, decoder, criterion, optimizer, epoch, args):
        encoder.train()
        decoder.train()

        device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = AverageMeter()
        total_step = len(train_data_loader)
        Avg_train_loss = np.zeros((1,total_step))

        for i, (images, scanpaths, lengths) in enumerate(train_data_loader):
            

            images = images.to(device)
            scanpaths = scanpaths.to(device)
            packed_targets = pack_padded_sequence(scanpaths, lengths, batch_first=True)
            targets = pack_padded_sequence(scanpaths, lengths, batch_first=True)[0]
            features = encoder(images)

            outputs, unpacked_outputs, lens = decoder(features, scanpaths, lengths)
            loss = criterion(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.data)

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, train_loss.avg, np.exp(loss.item()))) 
                
            Avg_train_loss[0,i] = train_loss.avg.cpu().numpy()
        
        snapshot_name = 'epoch_%d' % (epoch)
        np.save('Avg_train_loss'+ snapshot_name, Avg_train_loss)    

def classidtomap(unpacked_outputs, predicted, last_pts, lens, args):
        
        vocab_size = args.feat_size*args.feat_size+2
        unpacked_outputs1 = torch.zeros([args.batch_size, lens,vocab_size])
        unpacked_outputs1 = unpacked_outputs1.to(device)
        maps = torch.zeros([args.batch_size, lens,vocab_size])
        maps = maps.to(device)
        for batch_id in range(0,args.batch_size):
            unpacked_outputs1[batch_id][0:int(last_pts[batch_id])][:] = unpacked_outputs[batch_id][0:int(last_pts[batch_id])][:]
            for j in range(0,int(last_pts[batch_id])):
                maps[batch_id][j][predicted[batch_id][j]] = (unpacked_outputs1[batch_id][j][predicted[batch_id][j]])/(unpacked_outputs1[batch_id][j][predicted[batch_id][j]])

        maps = torch.narrow(maps,2,1,vocab_size-2)
        maps = torch.sum(maps,1)
        maps = maps.reshape([args.batch_size,args.feat_size,args.feat_size])

        return maps

def lastpoints(predicted, feat_size):
        
        seq_len = feat_size*feat_size
        end_tk = seq_len+1
        batch_size = predicted.shape[0]
        
        last_pts = torch.zeros([batch_size])
        last_pts = last_pts.to(device)
        for batch_id in range(0,batch_size):
            traj = predicted[batch_id]
            if len((traj == end_tk).nonzero())> 0:
                last_pt = (traj == end_tk).nonzero()[0]
            else:
                last_pt = len(traj)
            last_pts[batch_id]=last_pt
        return last_pts

def validate(val_data_loader, encoder, decoder, optimizer, epoch, args):
        encoder.eval()
        decoder.eval()

        device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
        val_loss = AverageMeter()
        total_step = len(val_data_loader)

        class_seqID_ll = np.zeros((1,total_step))
        classID = np.zeros([512,512])

        snapshot_name = 'epoch_%d' % (epoch)
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder' + snapshot_name + '.ckpt'))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder' + snapshot_name + '.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--image_resize', type=int, default=256 , help='input image to be resized to')
    parser.add_argument('--encoder_path', type=str, default='models/encoderepoch_200.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoderepoch_200.ckpt', help='path for trained decoder')
    parser.add_argument('--train_path', type=str, default='data/train/', help='path for train annotation json file')
    parser.add_argument('--val_path', type=str, default='data/val/', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=100, help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=600, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--feat_size', type=int, default=16)
    args = parser.parse_args()
    print(args)
    main(args)
