from torchvision.transforms.transforms import ColorJitter, RandomRotation, RandomVerticalFlip
from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
import pathlib
from torchvision.io import read_image
import numpy as np 
import cv2 # OpenCV
import os
import torch

#import mahotas
#import dlib
#import pcl   #complex dependencies required

# create dataset class
class knifeDataset(Dataset):
    def __init__(self,images_df,mode="train"):
        self.images_df = images_df.copy()
        self.images_df.Id = self.images_df.Id
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        # Read the image using the full path
        X,fname = self.read_images(index)
        labels = self.images_df.iloc[index].Label

        #if not self.mode == "test":
            #labels = self.images_df.iloc[index].Label
        #else:
            #labels = str(self.images_df.iloc[index].Id.absolute().__str__())
            #labels = None

        if self.mode == "train":
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight, config.img_height)),
                    T.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
                    T.RandomRotation(degrees=(0, 180)),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        elif self.mode == "val":
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight, config.img_height)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        elif self.mode == "test":
            X = T.Compose([T.ToPILImage(),
                    # Ensure all images have the same dimensions
                    T.Resize((config.img_weight, config.img_height)),
                    # Convert the images to PyTorch tensors for processing with PyTorch models.
                    T.ToTensor(),
                    # Normalize the images based on the *mean* and *standard deviation*. This is done
                    # to align the data distribution with the distribution used during the pretraining 
                    # of the model (often on datasets like ImageNet). 
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)

        return X.float(), labels, fname
        #return torch.from_numpy(X).float(), labels, fname

    def read_images(self,index):
        row = self.images_df.iloc[index]          # Returns the integer of Image Label
        #print("...................................................................................................")
        #print(row)
        #print("ROW ID=", row.Id)

        base_path = "/content/drive/My Drive/Knives"       # Remove './' from the beginning
        #relative_path = str(row.Id).lstrip('./')            # row'Id' column contains the relative paths
        #relative_path = str(row.Id)
        #filename = os.path.join(base_path, relative_path)  # Prepend base path to the relative path
        filename = str(row.Id)

        #print("Relative PATH=", relative_path)
        #print("***********************************************************************************")          
        #print("Full Path:", filename)
        
        # Convert BGR to RGB if the image was successfully read
        #im = cv2.imread(relative_path)[:,:,::-1]
        #im = cv2.imread(filename)[:,:,::-1] if cv2.imread(filename) is not None else None
        im = cv2.imread(filename)  # Attempt to read the image once
        if im is not None:
            im = im[:, :, ::-1]  # Convert from BGR to RGB if image read successfully
        else:
            print(f"Failed to read image at path: {filename}")

        return im, filename


