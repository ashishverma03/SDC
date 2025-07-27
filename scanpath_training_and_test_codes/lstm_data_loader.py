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
        scan_basename = self.list_sample[index]

        path_scanpaths = os.path.join(self.root_scan, scan_basename)
        path_img = os.path.join(self.root_img, scan_basename[:-4]+'.jpg')

        assert os.path.exists(path_scanpaths), '[{}] does not exist'.format(path_scanpaths)
        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)

        image = Image.open(path_img)
        org_image_sz = image.size[0]
        image = image.resize([self.image_resize, self.image_resize], Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)
        
        scanpaths = np.load(path_scanpaths)
        scanpaths = np.transpose(scanpaths)
        scanpaths = self.FixationtoclassID(scanpaths, 256, self.feat_sz)
        scanpaths = np.squeeze(scanpaths.astype(int))
        
        target = torch.from_numpy(scanpaths)

        return image, target

    def __len__(self):
        return len(self.list_sample)

    def FixationtoclassID(self, traj, org_img_sz, feat_sz):

        traj = traj[:,traj.min(axis=0)>=0]

        div_factor = org_img_sz/feat_sz
        regionID_seq = []
        perv_region_id = 10000
        for pt in range(len(traj[0])):
            
            px = traj[0,pt]
            py = traj[1,pt]

            if 0<px<257 and 0<py<257:
             
                m = np.ceil(px/div_factor)
                n = np.ceil(py/div_factor)
                region_id = m+(n-1)*feat_sz
                if region_id != perv_region_id:
                    regionID_seq = np.append(regionID_seq, region_id)
                    perv_region_id = region_id

        end_token = feat_sz*feat_sz+1
        regionID_seq = np.append(regionID_seq, end_token)
        regionID_seq = regionID_seq[regionID_seq<258]
        return regionID_seq


def collate_fn(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, scanpaths = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(scan) for scan in scanpaths]
    targets = torch.zeros(len(scanpaths), max(lengths)).long()
    for i, scan in enumerate(scanpaths):
        end = lengths[i]
        targets[i, :end] = scan[:end]        
    return images, targets, lengths

def get_loader(image_resize, feat_sz, data_path, transform, batch_size, shuffle, num_workers):

    traj_data = TrajDataset(image_resize, feat_sz, data_path=data_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=traj_data, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
