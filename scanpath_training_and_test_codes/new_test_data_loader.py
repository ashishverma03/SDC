import torch
import glob
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import pickle
import numpy as np
import pdb
from PIL import Image

def getFileNamesFromFolder(input_path, tag):

    files = []
    for path in glob.glob(input_path + "*{}".format(tag)):
        files.append(os.path.basename(path))

    files = sorted(files)
    
    return files

class TrajDataset(data.Dataset):

    def __init__(self, image_resize, feat_sz, data_path, transform=None, is_train=1):     
        self.root_img = data_path
        self.root_scan = data_path
        self.is_train = is_train
        self.transform = transform
        self.image_resize = image_resize
        self.feat_sz = feat_sz


        files_captions = getFileNamesFromFolder(self.root_img, ".png")

        self.list_sample = sorted(files_captions)


    def __getitem__(self, index):
        image_basename = self.list_sample[index]

        path_img = os.path.join(self.root_img, image_basename)

        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)

        image = Image.open(path_img)
        org_image_sz = image.size[0]
        image = image.resize([self.image_resize, self.image_resize], Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)


        return image, image_basename

    def __len__(self):
        return len(self.list_sample)


def get_loader(image_resize, feat_sz, data_path, transform):

    traj_data = TrajDataset(image_resize, feat_sz, data_path=data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=traj_data)
    return data_loader
