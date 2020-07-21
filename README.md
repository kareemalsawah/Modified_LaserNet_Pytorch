# Simplified LaserNet

This is an implementation of an idea that simplifies the LaserNet paper for the task of detecting cones around a track using lidar clouds and RGB images. The simplified method is based on clustering methods. The CNN segments the lidar cloud; then, a method used to cluster the points of each class efficiently was created based on the idea of NMS. Finally, K-means is used on the found points to shift the points more towards the cone center.

## How to use this repository
### 1) Downloading the dataset
The dataset could be downloaded from [here](https://drive.google.com/file/d/1Dr_ILGfhudFuJpiO5nE-x04aImWVeRJW/view?usp=sharing). Unzip the saved file so that you have a folder data with images.npy, lidar.npy, and targets.npy in it.

### 2) Training
Training is done using the command

  python train.py

Parameters:
* --epochs: Number of epochs
* --batch_size: Batch size
* --lr: learning rate
* --lidar_data: path to the lidar data npy
* --image_data: path to rgb data npy
* --targets: path to targets npy
* --train_ratio: a value from 0 to 1 where 1 means all data will be used from training
