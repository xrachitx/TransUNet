import os
import numpy as np
import pandas as pd
from torchvision import transforms
import skimage.io as io
import skimage
from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class LoadData(Dataset):
    def __init__(self, fileNames, rootDir, double_channel = False,transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.dc = double_channel
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0][1:])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1][1:])
        print(inputName,targetName,self.rootDir, self.frame.iloc[idx, 0])
        inputImage = cv2.imread(inputName)
        targetImage = cv2.imread(targetName, cv2.IMREAD_GRAYSCALE)
        targetImage = targetImage > 0.0
        print(targetImage)
        print(np.unique(targetImage,return_counts=True))
        print("first counts")
        if self.transform:
            inputImage = Image.fromarray(inputImage)
            inputImage = self.transform(inputImage)
            inputImage = np.array(inputImage)
            targetImage = Image.fromarray(targetImage)
            targetImage = self.transform(targetImage)
            targetImage = np.array(targetImage)
        if self.dc:
            out_im = np.zeros((448,448,2))
            out_im[:,:,0] = np.where(targetImage == False, 1, 0)
            out_im[:,:,1] = np.where(targetImage == True, 1, 0)
            out_im = out_im.transpose((2, 0, 1))
            targetImage = out_im
        else: 
            targetImage = np.expand_dims(targetImage,axis=0)
        counts = np.unique(targetImage,return_counts=True)[1]
        print(targetImage)
        print(counts)
        print("counts")
        weights = np.array([ counts[0]/(counts[0]+counts[1]) , counts[1]/(counts[0]+counts[1]) ])
        inputImage = inputImage.astype(np.float32)
        targetImage = targetImage.astype(np.float32)
        inputImage = inputImage.transpose((2, 0, 1))
        
        print(inputImage.shape, targetImage.shape, weights.shape)
        return inputImage, targetImage,weights, self.frame.iloc[idx, 0]
        

if __name__ == "__main__":
    rootDir ="./CoSkel+"
    files = "./CoSkel+/train.csv"

    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=20)
    # print(train_dataloader)
    for i, (data) in enumerate(train_dataloader,0):
        print(data[0].shape,data[1].shape,data[2])
        exit()
    # print(len(train_dataloader))
