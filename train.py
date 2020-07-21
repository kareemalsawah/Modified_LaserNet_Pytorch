import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm
from models import LaserNet, Bounding_Box_Detector
import argparse
from dataset import LaserNet_Dataset
from torch.utils.data import DataLoader

def train(args):
    epochs = args.epochs
    lr = args.lr
    lidar_data_pth = args.lidar_data
    image_data_pth = args.image_data
    targets_pth = args.targets

    # Load data
    lidar_data = np.load(lidar_data_pth, mmap_mode='r')[2:]
    image_data = np.load(image_data_pth, mmap_mode='r')[2:]
    targets = np.load(targets_pth, mmap_mode='r')[2:]

    print("Loaded data, shuffling it now")

    # Shuffle data
    random_perm = np.random.permutation(lidar_data.shape[0])
    lidar_data = lidar_data[random_perm]
    image_data = image_data[random_perm]
    targets = targets[random_perm]

    # Print data shapes
    print("Lidar data shape: {}".format(lidar_data.shape))
    print("RGB Image data shape: {}".format(image_data.shape))
    print("Targets data shape: {}".format(targets.shape))

    # Creating the datasets and dataloaders
    num_training_images = int(lidar_data.shape[0]*args.train_ratio)
    train_dataset = LaserNet_Dataset(lidar_data[:num_training_images],image_data[:num_training_images],targets[:num_training_images])
    val_dataset = LaserNet_Dataset(lidar_data[num_training_images:],image_data[num_training_images:],targets[num_training_images:])
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    # Training code
    LN = LaserNet().cuda()
    optimizer = torch.optim.Adam(LN.parameters(),lr=lr)
    criterion = nn.NLLLoss()


    LN.train()
    start_time = time.time()
    for e in range(epochs):
      epoch_loss = 0
      for lidar_in,img_in,target_in in train_loader:
        lidar_in,img_in,target_in = lidar_in.cuda() ,img_in.cuda() ,target_in.cuda()

        class_probs = LN.forward(img_in,lidar_in)
        optimizer.zero_grad()
        loss = criterion(F.log_softmax(class_probs,dim=1),target_in)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
      if e%5 == 0 and e>0:
        checkpoint = LN.state_dict()
        torch.save(checkpoint, './weights/LaserNet_'+str(e)+'.pth')
      print("Epoch: {}, Loss: {}".format(e+1,epoch_loss/iters_per_batch))

    checkpoint = LN.state_dict()
    torch.save(checkpoint, './weights/LaserNet_Last.pth')
    training_time = time.time() - start_time
    print("Training time: {:.2f} minutes".format(training_time/60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the LaserNet network")
    parser.add_argument('--epochs',default=20,type=int,help='Number of epochs to train the network')
    parser.add_argument('--batch_size',default=8,type=int,help='Batch size to use during training')
    parser.add_argument('--lr',default=0.0015,type=float,help='Optimizer Learning Rate')
    parser.add_argument('--lidar_data',default='./data/lidar_data.npy',type=str,help='Lidar data as an .npy file to use for training')
    parser.add_argument('--image_data',default='./data/images.npy',type=str,help='RGB image data as an .npy file to use for training')
    parser.add_argument('--targets',default='./data/targets.npy',type=str,help='targets as an .npy file to use for training')
    parser.add_argument('--train_ratio',default=0.8,type=float,help='Ratio used for training for data (1.0 means all data will be used for training)')

    args = parser.parse_args()
    train(args)
