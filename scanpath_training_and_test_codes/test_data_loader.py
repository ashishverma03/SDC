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

        files_scanpaths = getFileNamesFromFolder(self.root_scan, ".npy")

        self.list_sample = sorted(files_scanpaths)


    def __getitem__(self, index):
        scanpath_basename = self.list_sample[index]
        print(scanpath_basename)

        path_scanpaths = os.path.join(self.root_scan, scanpath_basename)
        path_img = os.path.join(self.root_img, scanpath_basename[:-4]+'.jpg')

        assert os.path.exists(path_scanpaths), '[{}] does not exist'.format(path_scanpaths)
        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)

        image = Image.open(path_img)
        org_image_sz = image.size[0]
        image = image.resize([self.image_resize, self.image_resize], Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)
        
        scanpaths = np.load(path_scanpaths)
        scanpaths = np.transpose(scanpaths)
        scanpaths = np.squeeze(scanpaths.astype(int))
        
        target = torch.from_numpy(scanpaths)

        return image, target, scanpath_basename

    def __len__(self):
        return len(self.list_sample)

    def FixationtoclassID(self, traj, org_img_sz, feat_sz):

        traj = traj[:,traj.min(axis=0)>=0]

        div_factor = org_img_sz/feat_sz

        regionID_seq = np.zeros(len(traj[0])+1)

        p = 0
        for pt in range(len(traj[0])):
            
            px = traj[0,pt]
            py = traj[1,pt]
            if 0<px<257 and 0<py<257:
             
                m = np.ceil(px/div_factor)
                n = np.ceil(py/div_factor)
                region_id = m+(n-1)*feat_sz
                regionID_seq[p] = region_id 
                p = p+1
        regionID_seq[p] = feat_sz*feat_sz+1
        regionID_seq = regionID_seq[regionID_seq<258]
        return regionID_seq


def get_loader(image_resize, feat_sz, data_path, transform):

    traj_data = TrajDataset(image_resize, feat_sz, data_path=data_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=traj_data)
    return data_loader
