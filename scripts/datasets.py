import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import skimage
import torch
import numpy as np
    
# Step 1: Define your custom dataset class by inheriting from torch.utils.data.Dataset
class ImageDataset(Dataset):    
    def __init__(self, image_dir, labels, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            labels (dict): A dictionary mapping image filenames to labels.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir          # Save the path to image directory
        self.labels = labels                # Save the labels dictionary
        self.transform = transform          # Save the transformation pipeline
        self.image_filenames = list(labels.keys())  # Store all image filenames

    def __len__(self):
        # Return the total number of samples
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image and label at the given index
        img_name = self.image_filenames[idx]
        
        # Load the image and convert to grayscale for MRI 
        img_path_mri = os.path.join(self.image_dir, img_name)
        image_mri = Image.open(img_path_mri).convert("L")  
        
        # Load the image and convert RGB for Cryo
        img_path_cryo = os.path.join(self.image_dir, img_name)
        image_cryo = Image.open(img_path_cryo).convert("RGB")
        
        # Load the label
        img_path_label = os.path.join(self.image_dir, img_name)
        label_map = Image.open(img_path_label)
        
        # Apply transformation if specified
        if self.transform:
            image_mri = self.transform(image_mri)
        
        

        return {'A':image_mri, 'B': image_cryo, 'C':label_map}

