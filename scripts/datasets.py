import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import skimage
import torch
import numpy as np

    


uniques = len([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       52])



def get_channel_mask(img, uniques):
    class_neglected = 13
    img_relabeled = np.where(img>class_neglected, img-1, img)
    channel_vol = np.zeros((uniques, 256, 256)).astype(np.uint8)
    channeled_mask = np.zeros((256,256)).astype(np.uint8)

    for i in range(uniques):
        channel_wise_boolarray = np.where(img_relabeled==i, channeled_mask+1, channeled_mask)
        channel_vol[i, ...] = channel_wise_boolarray
    
    return channel_vol
        



class ImageDataset(Dataset):
    def __init__(self, files, transform, mode, n_channels=54):
        self.files = files
        self.transform = transform
        self.mode = mode
        self.n_channels = n_channels
    
    def __getitem__(self, idx):
        # img_seg_singleclass = self.transform(Image.open('{}{}/seg/{}'.format(self.data_dir, self.mode, self.files[idx])))
        img_seg_singleclass = torch.from_numpy(skimage.io.imread('{}/seg/{}'.format(self.files[idx][0], self.files[idx][1])))
        try:
            img_seg_singleclass = img_seg_singleclass[:256, :256].view(1, 256, 256)
        except Exception as e:
            print("Incorrect shape  ", '{}/seg/{}'.format(self.files[idx][0], self.files[idx][1]))
        
        img_mri= self.transform(Image.open('{}/mri/{}'.format(self.files[idx][0], self.files[idx][1])))[..., :256, :256]#*255
        img_seg =  torch.from_numpy(get_channel_mask(img_seg_singleclass, uniques))
        img_cryo= self.transform(Image.open('{}/cryo/{}'.format(self.files[idx][0], self.files[idx][1])))[..., :256, :256]#*255
        return  {'A':img_mri, 'B': img_cryo, 'C':img_seg} 
    def __len__(self):
        return len(self.files)
    

