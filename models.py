import numpy as np
import torch
from torch import nn
import time
import torch.nn.functional as F
from tqdm import tqdm
import cupy as cp
from utils import *

BatchNorm = nn.BatchNorm2d
'''Basic Resnet Block
Conv2d
BN
Relu
Conv2d
BN
Relu

with residual connection between input and output
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.stride = stride
        self.project = None
        if in_channels!=out_channels:
            self.project = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.project != None:
            residual = self.project(residual)
        out += residual
        out = self.relu(out)

        return out

'''
Deconvolution Layer for upsampling
TransposeConv2d
BN
Relu
'''
class Deconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,padding=0)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


'''
Feature Aggregator Module described in the LaserNet paper
'''
class Feature_Aggregator(nn.Module):
    def __init__(self,in_channels_1,in_channels_2,out_channels):
        super().__init__()
        self.deconv = Deconv(in_channels_2,out_channels)
        self.block_1 = BasicBlock(in_channels_1+in_channels_2,out_channels)
        self.block_2 = BasicBlock(out_channels,out_channels)

    def forward(self,x1,x2):
        x2 = self.deconv(x2)
        x1 = torch.cat([x1,x2],1)
        x1 = self.block_1(x1)
        x1 = self.block_2(x1)
        return x1

'''
DownSample module using Conv2d with stride > 1
Conv2d(stride>1)
BN
Relu
Conv2d
BN
Relu
'''
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=2, padding=dilation,
                                   bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=dilation,
                                bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.project = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.project(residual)
        out += residual
        out = self.relu(out)
        return out

'''
Feature Extrator module described in LaserNet paper
DownSample input if not 1a
'''
class Feature_Extractor(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks=6,down_sample_input=False):
        super().__init__()
        self.down_sample = None
        self.down_sample_input = down_sample_input
        if down_sample_input:
            self.down_sample = DownSample(in_channels,out_channels)

        blocks_modules = []
        for i in range(num_blocks):
            if i == 0 and not down_sample_input:
                blocks_modules.append(BasicBlock(in_channels,out_channels))
            else:
                blocks_modules.append(BasicBlock(out_channels,out_channels))
        self.blocks = nn.Sequential(*blocks_modules)

    def forward(self,x):
        if self.down_sample_input:
            x = self.down_sample(x)
        x = self.blocks(x)
        return x

'''
Main Deep Aggregation class described as in LaserNet paper
num_outputs is the number of channels of the output image
output image has the same width and height as input image
'''
class Deep_Aggregation(nn.Module):
    def __init__(self,num_inputs,channels,num_outputs):
        super().__init__()
        self.extract_1a = Feature_Extractor(num_inputs,channels[0])
        self.extract_2a = Feature_Extractor(channels[0],channels[1],down_sample_input=True)
        self.extract_3a = Feature_Extractor(channels[1],channels[2],down_sample_input=True)
        self.aggregate_1b = Feature_Aggregator(channels[0],channels[1],channels[1])
        self.aggregate_1c = Feature_Aggregator(channels[1],channels[2],channels[2])
        self.aggregate_2b = Feature_Aggregator(channels[1],channels[2],channels[2])
        self.conv_1x1 = nn.Conv2d(channels[2],num_outputs,kernel_size=1,stride=1)

    def forward(self,x):
        x_1a = self.extract_1a(x)
        x_2a = self.extract_2a(x_1a)
        x_3a = self.extract_3a(x_2a)
        x_1b = self.aggregate_1b(x_1a,x_2a)
        x_2b = self.aggregate_2b(x_2a,x_3a)
        x_1c = self.aggregate_1c(x_1b,x_2b)
        out = self.conv_1x1(x_1c)
        return out

# Note to self, can make strides modify size along the horizontal axis only

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels_num,dim_change=True,custom_stride=(2,2)):
        super().__init__()
        self.in_channels = in_channels
        self.channels_num = channels_num
        self.dim_change = dim_change
        self.resUnit1 = ResUnit(self.in_channels, channels_num = self.channels_num, filter_size=3, dim_change=dim_change,custom_stride=custom_stride)
        self.resUnit2 = ResUnit(self.channels_num, channels_num = self.channels_num, filter_size=3, dim_change=False)
        self.resUnit3 = ResUnit(self.channels_num, channels_num = self.channels_num, filter_size=3, dim_change=False)
        self.resUnit4 = ResUnit(self.channels_num, channels_num = self.channels_num,filter_size=3, dim_change=False)
        if self.dim_change:
            self.reshaping_conv = nn.Conv2d(self.in_channels, self.channels_num, 1, stride=custom_stride, padding= 0) # 1 x 1 conv on the residual connection to change dimension of the residue
        else:
            self.reshaping_conv = nn.Conv2d(self.in_channels, self.channels_num, 1, stride=1, padding= 0) # 1 x 1 conv on the residual connection to change dimension of the residue

    def forward(self, x):
        residue = self.reshaping_conv(x)
        x = self.resUnit1(x)
        x = x + residue
        residue = x
        x = self.resUnit2(x)
        x = x + residue
        residue = x
        x = self.resUnit3(x)
        x = x + residue
        residue = x
        x = self.resUnit4(x)
        x = x + residue
        return x

class ResUnit(nn.Module):
    def __init__(self, in_channels, channels_num, filter_size = 3, dim_change = False,custom_stride=(2,2)):
        super().__init__()
        self.stride = 1
        if dim_change:
            self.stride = custom_stride
        self.conv1 = nn.Conv2d(in_channels, channels_num, filter_size, stride = self.stride, padding = 1)
        self.conv2 = nn.Conv2d(channels_num, channels_num, filter_size, stride = 1, padding = 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class AuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resBlock1 = ResBlock(in_channels = 3, channels_num =16,dim_change=True)
        self.resBlock2 = ResBlock(in_channels = 16, channels_num = 24,dim_change=True,custom_stride=(1,2))
        self.resBlock3 = ResBlock(in_channels = 24, channels_num = 32,dim_change=True,custom_stride=(1,2))

    def forward(self, x):
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        return x

'''
LaserNet network as described in the original paper, created using the Deep Aggregation Network.
The fusion between RGB and Lidar used is similar to the one in the LaserNet++ paper.
'''
class LaserNet(nn.Module):
    def __init__(self,deep_aggregation_num_channels=[64,128,128],num_out_channels=4):
        super().__init__()
        self.RGB_CNN = AuxNet()
        self.DL = Deep_Aggregation(64,deep_aggregation_num_channels,num_out_channels)
        self.Lidar_CNN = nn.Conv2d(6,32,kernel_size=3,padding=1)

    def forward(self,rgb_img,lidar):
        rgb_semantics = self.RGB_CNN.forward(rgb_img)
        lidar_semantics = self.Lidar_CNN(lidar)
        fused_semantics = torch.cat((rgb_semantics,lidar_semantics),dim=1)
        out = self.DL.forward(fused_semantics)
        return out

'''
Radius NMS class used to execute the radius nms taking the LaserNet segmentation output as input and returing the reduced boxes
'''
class Radius_NMS(nn.Module):
  def __init__(self,num_classes,min_height,max_height,radius=3):
    super().__init__()
    self.num_classes = num_classes
    self.min_height = min_height
    self.max_height = max_height
    self.radius = radius
  def forward(self,segmented_channels,lidar):
    val,ind = segmented_channels.max(dim=1)
    batch_size = segmented_channels.shape[0]
    Lidar_W = lidar.shape[2]
    Lidar_H = lidar.shape[3]
    t = torch.tensor([1]).cuda()

    s = time.time()
    results = []
    for b in range(batch_size):
      class_res = []
      for c in range(1,self.num_classes):
        class_cond = ind[b]==c

        lidar_points_x = lidar[b][0][class_cond].reshape(-1,1)
        lidar_points_y = lidar[b][1][class_cond].reshape(-1,1)

        preds = torch.cat((lidar_points_x,lidar_points_y),dim=1).cuda()
        if preds.shape[0] > 0:
          res = radius_nms_cp(preds.detach().cpu().numpy(),self.radius)
          class_res.append(preds[res])
        else:
          class_res.append([])
      results.append(class_res)
    time_taken = time.time()-s
    return results,time_taken

'''
Complete model including LaserNet and radius nms, a forward pass would take Lidar and RGB as input and return the predicted boxes.
K-means is an option to use to improve the box center predictions.
'''
class Bounding_Box_Detector(nn.Module):
  def __init__(self,LN_model,num_out_channels=4,min_height=0,max_height=2.5,box_l=1,box_w=1,k_means_iters=1,radius=3):
    super().__init__()
    self.LN = LN_model
    self.num_classes = num_out_channels
    self.NMS = Radius_NMS(num_out_channels,min_height,max_height,radius).cuda()
    if use_cuda:
      self.LN = self.LN.cuda()
    self.k_means_iters = k_means_iters
    self.only_NN = False
    self.kmeans = True
    self.radius = radius
    self.num_classes = num_out_channels

  def forward(self,rgb,lidar,verbose=False):
    times = [] # Used for testing purposes
    start = time.time()

    # Forward pass through LaserNet
    batch_size = rgb.shape[0]
    lidar_copy = lidar.clone()
    NN_result = self.LN.forward(rgb,lidar)
    NN_time = time.time()-start
    if verbose:
      print("NN time: {} ms".format(NN_time*1000))
      times.append(NN_time*1000)

    # Only_NN used to skip nms step for training purposes
    if self.only_NN:
      return NN_result


    # Apply SoftNMS
    start = time.time()
    box_predictions,new_time = self.NMS.forward(NN_result.detach().cpu(),lidar_copy.detach().cpu())
    SNMS_time = time.time()-start
    if verbose:
      print("SoftNMS time: {} ms".format(new_time*1000))
      times.append(new_time*1000)

    # Refine Predictions using K-means
    k_means_predictions = []
    if self.kmeans:
      start = time.time()
      val,ind = NN_result.max(dim=1)
      for b in range(batch_size):
        class_preds = []
        for c in range(1,self.num_classes):
          boxes_tensor = box_predictions[b][c-1]
          if (not type(boxes_tensor) == type([])) and boxes_tensor.shape[0]>0:
            start_preds = boxes_tensor
            num_of_cones = start_preds.shape[0]
            class_cond = ind[b]==c
            lidar_points_x = lidar[b][0][class_cond].reshape(-1,1)
            lidar_points_y = lidar[b][1][class_cond].reshape(-1,1)

            class_lidar_points = torch.cat((lidar_points_x,lidar_points_y),dim=1)
            cl, k_preds = kmeans(X=class_lidar_points,cluster_centers=start_preds,num_clusters=num_of_cones,iters=self.k_means_iters)
            class_preds.append(k_preds)
          else:
            class_preds.append([])
        k_means_predictions.append(class_preds)
      box_predictions = k_means_predictions
      k_means_time = time.time()-start
      if verbose:
        print("K-means refinement time: {} ms".format(k_means_time*1000))
        times.append(k_means_time*1000)
    return box_predictions,times
