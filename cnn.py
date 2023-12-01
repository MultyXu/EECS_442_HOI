import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# TODO: Create your own network
def make_layers(batch_norm=True):
  # VGG for object recognition 
  cfg = [64, 'M', 128, 'M', 128, 128, 'M']
  layer_dic = []
  prev_channel = 0
  new_channel = 3
  for i in cfg:
    if type(i) is int:
      prev_channel = new_channel
      new_channel = i
      # construct 2d Convolution layer
      conv2d = nn.Conv2d(prev_channel, new_channel, (3,3), (1,1), (1,1))
      layer_dic.append(conv2d)
      # if need to batch_nor
      if batch_norm:
        batch_norm = nn.BatchNorm2d(new_channel, 1e-05, 0.1, True, True)
        layer_dic.append(batch_norm)
      # construct relue
      layer_dic.append(nn.ReLU(inplace=True))
    elif i == 'M':
      #construct
      max_pool = nn.MaxPool2d(2, 2, 0,1, ceil_mode=False)
      layer_dic.append(max_pool)

    # * means unpack the list
    features = nn.Sequential(*layer_dic)

    return features

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = make_layers()
    self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

    layer_dic = []
    layer_dic.append(nn.Linear(1600, 512, True))
    layer_dic.append(nn.ReLU(True))
    layer_dic.append(nn.Dropout(0.3))
    layer_dic.append(nn.Linear(512, 256, True))
    layer_dic.append(nn.ReLU(True))
    layer_dic.append(nn.Dropout(0.3))
    layer_dic.append(nn.Linear(256, 2, True))

    self.proposer = nn.Sequential(*layer_dic)

    # initialize the weight
    self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    print(x.shape)
    x = torch.flatten(x, 1)
    x = self.proposer(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)