import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pdb


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, embed_size+256)  
        self.bn = nn.BatchNorm1d(embed_size+256, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, feat_size, num_layers, max_seq_length=200):
        super(DecoderRNN, self).__init__()

        self.vocab_size = feat_size*feat_size+2
        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size+256, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.linear2 = nn.Linear(embed_size+256, 256)  # for image feature size reduction
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):

        features1 = self.linear2(features)
        embeddings = self.embed(captions)
        features1 = features1.unsqueeze(1).repeat(1,embeddings.size()[1],1)
        embeddings = torch.cat((features1, embeddings), 2)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, (ht, ct) = self.lstm(packed)
        
        outputs = self.linear(hiddens[0])

        unpacked_hiddens,lens = pad_packed_sequence(hiddens, batch_first=True)
        unpacked_outputs = self.linear(unpacked_hiddens)
        return outputs, unpacked_outputs, lens
    
    def sample(self, features, states=None):
        sampled_ids = []
        features1 = self.linear2(features)
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))           
            _, predicted = outputs.max(1)             
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                 
            inputs = torch.cat((features1, inputs), 1)
            inputs = inputs.unsqueeze(1)            

        sampled_ids = torch.stack(sampled_ids, 1)     
        return sampled_ids

    def sample_train(self, features, states=None):
        sampled_ids = []
        
        features1 = self.linear2(features)
        features2 = features1.unsqueeze(1)
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens)            
            _, predicted = outputs.max(2)                   
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)           
            inputs = torch.cat((features2, inputs), 2)

            if i==0:
                outputs1 = outputs
            else:
                outputs1 = torch.cat((outputs1,outputs),1)
        
        sampled_ids = torch.stack(sampled_ids, 1)       
        return outputs1,sampled_ids



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count