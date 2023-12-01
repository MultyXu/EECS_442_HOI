import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from imageio.v3 import imread
from torch.utils.data import Dataset, DataLoader
import __init__
# import vsrl_utils as vu
# from coco.PythonAPI.pycocotools import coco
import vcoco.vsrl_utils as vu
from vcoco.coco.PythonAPI.pycocotools import coco
from PIL import Image
import requests

from tqdm import tqdm
from imageio.v3 import imread

def get_train_val_test_loaders(batch_size):
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the LandmarksDataset constructor.
    """
    tr, te = get_train_val_test_datasets()

    #tr, va, te, _ = get_train_val_test_datasets()

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=False)
    # va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader
    #return tr_loader, va_loader, te_loader

def get_train_val_test_datasets():
    """Return LandmarksDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr = V_COCO("test", 0, 300)
    # va = V_COCO("val")
    te = V_COCO("test", 300, 330)

    # return tr, va, te
    return tr, te

class V_COCO(Dataset):
    """Dataset class for landmark images."""

    def __init__(self, partition="train", start=0, end=100):    
        
        super().__init__()
        
        self.partition = partition
        self._load_cocos(partition)
        self.start = start
        self.end = end
        self.X, self.y = self._load_data()

    def __len__(self):
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx, include_object=False):
        """Return (image, label) pair at index `idx` of dataset."""
        return (torch.from_numpy(self.X[idx]).permute(2,0,1) / 255), torch.tensor(self.y[idx]/640).float()
        # return self.X[idx], self.y[idx]

    def _load_cocos(self, set_name):
        
        # Load COCO annotations for V-COCO images
        self.coco = vu.load_coco()

        # Load the VCOCO annotations for vcoco_train image set
        self.vcoco_all = vu.load_vcoco('vcoco_' + set_name)
        for x in self.vcoco_all:
            x = vu.attach_gt_boxes(x, self.coco)
            
    def _load_data(self):
        X, y = [], []
        classes = [x['action_name'] for x in self.vcoco_all]

        for i in range(5):
            if i != 0:
              # only load hold action
              break
            # for each action
            action_name = self.vcoco_all[i]['action_name']
            print("load " + action_name, i+1, "/", len(self.vcoco_all))
            cls_id = classes.index(action_name)
            vcoco = self.vcoco_all[cls_id]

            positive_index = np.where(vcoco['label'] == 1)[0]
            # if want image to be random order
            # positive_index = np.random.permutation(positive_index)
            
            # for i in tqdm(range(len(positive_index))):
            # for i in tqdm(range(100)): #just load first 100 images

            # load_num = 100
            print('total', self.partition, 'image number', len(positive_index))

            # if len(positive_index) < load_num:
            #   load_num = len(positive_index)
            # for j in tqdm(range(load_num)): #just load first 100 images

            self.end = min(self.end, len(positive_index))
            print('load image from ', self.start, 'to', self.end)
            for j in tqdm(range(self.start, self.end)):
                id = positive_index[j]
                #X:

                
                vcoco_image = self.coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
                # print(vcoco_image)
                vcoco_image_url = vcoco_image['coco_url']
                # print(vcoco_image_url)
                # file = Image.open(requests.get(vcoco_image_url, stream=True).raw)
                # Image.open() is hard to convert to tensor, use imread instead
                # see 445 project 2 for more detail
                img = imread(vcoco_image_url)
                # naive solution: pad every image to 640, 640

                pad = np.zeros((640,640,3))
                pad[:img.shape[0], :img.shape[1], :] = img
                
                sy = 4.; sx = float(pad.shape[1])/float(pad.shape[0])*sy;
                # fig, ax = subplot(plt, 1,1, sy,sx); ax.set_axis_off();
                # ax.imshow(pad);

                X.append(pad)

                #y:
                role_object_id = vcoco['role_object_id'][id]
                x_cord = -500
                y_cord = -500

                # get role_box, 
                role_bbox = vcoco['role_bbox'][id,:]*1.
                role_bbox = role_bbox.reshape((-1,4))

                if len(role_bbox) > 2:
                  if not np.isnan(role_bbox[2,0]):
                      # bbox is actually "xyxy" format
                      x_cord = (role_bbox[2,0] + role_bbox[2,2]) / 2
                      y_cord = (role_bbox[2,1] + role_bbox[2,3]) / 2

                if len(role_bbox) > 1:
                  if not np.isnan(role_bbox[1,0]):
                      # bbox is actually "xyxy" format
                      x_cord = (role_bbox[1,0] + role_bbox[1,2]) / 2
                      y_cord = (role_bbox[1,1] + role_bbox[1,3]) / 2
                  
                    
                y.append((x_cord, y_cord))
        return np.array(X), np.array(y)