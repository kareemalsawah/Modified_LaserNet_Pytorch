import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class LaserNet_Dataset(Dataset):
  """LaserNet dataset."""

  def __init__(self, lidar_data, image_data, targets, transform=None):
      """
      Args:
          lidar_data: path to lidar data
          image_data: path to rgb image data
          targets: path to targets data
          transforms: optional transforms to apply to the rgb images
      """
      self.lidar_data = lidar_data
      self.image_data = image_data
      self.targets = targets
      self.transform = transform

  def __len__(self):
      return self.lidar_data.shape[0]

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      lidar = self.lidar_data[idx]
      image = self.image_data[idx]
      targets = self.targets[idx]

      lidar = torch.tensor(lidar).type(torch.FloatTensor)
      image = torch.tensor(image/255).type(torch.FloatTensor)
      targets = torch.tensor(targets).type(torch.LongTensor)[0]

      if self.transform:
        image = self.transform(image)

      return lidar,image,targets
