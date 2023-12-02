import os
import random
import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
import torch
from matplotlib import pyplot as plt
from imageio.v3 import imread
from torch.utils.data import Dataset, DataLoader
import vcoco.__init__
# import vsrl_utils as vu
# from coco.PythonAPI.pycocotools import coco
import vcoco.vsrl_utils as vu
from vcoco.coco.PythonAPI.pycocotools import coco
from PIL import Image
import requests

from tqdm import tqdm
from imageio.v3 import imread

def get_train_val_test_loaders(batch_size, train_num=100, test_num=20):
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the LandmarksDataset constructor.
    """
    tr, te = get_train_val_test_datasets(train_num, test_num)

    #tr, va, te, _ = get_train_val_test_datasets()

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=False)
    # va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader
    #return tr_loader, va_loader, te_loader

def get_train_val_test_datasets(train_num=100, test_num=20):
    """Return LandmarksDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr = V_COCO("test", 0, train_num)
    # va = V_COCO("val")
    te = V_COCO("test", train_num, train_num+test_num)

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
        self.X, self.y, self.original_image, self.human_center = self._load_data()

    def __len__(self):
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx, include_object=False):
        """Return (image, label) pair at index `idx` of dataset."""
        return torch.from_numpy(self.X[idx]).permute(2,0,1) / 255, torch.tensor(self.y[idx]).float(), torch.tensor(self.original_image[idx]/255).float(), torch.tensor(self.human_center[idx]).float()
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
        original_image = []
        human_center = []
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

            print('total', self.partition, 'image number', len(positive_index))

            self.end = min(self.end, len(positive_index))
            print('load image from ', self.start, 'to', self.end)
            for j in tqdm(range(self.start, self.end)):
                id = positive_index[j]
                #X:

                #get image url
                vcoco_image = self.coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
                vcoco_image_url = vcoco_image['coco_url']
                # print(vcoco_image_url)

                # get image from url
                img = Image.open(requests.get(vcoco_image_url, stream=True).raw)

                # naive solution: pad every image to 640, 640
                np_img = np.array(img)
                pad_image = np.zeros((640,640,3))
                pad_image[:np_img.shape[0], :np_img.shape[1], :] = np_img
                original_image.append(pad_image) 

                # crop out human and resize to 244 x 244
                bbox = vcoco['bbox'][[id],:][0]
                X1,Y1,X2,Y2 = bbox
                human_center_point = [(X1+X2)/2, (Y1+Y2)/2]
                cropped_img = img.crop((int(X1), int(Y1), int(X2), int(Y2)))
                resized_img = cropped_img.resize((224, 224))
                resized_img = np.array(resized_img)
                X.append(resized_img)
                
                #y:
                role_object_id = vcoco['role_object_id'][id]
                x_cord = -500
                y_cord = -500

                # get role_box, 
                role_bbox = vcoco['role_bbox'][id,:]*1.
                role_bbox = role_bbox.reshape((-1,4))

                # construct point of interest relative to the human center
                if len(role_bbox) > 1:
                  if not np.isnan(role_bbox[1,0]):
                      # bbox is actually "xyxy" format
                      # object center relative to human center
                      x_cord = (role_bbox[1,0] + role_bbox[1,2]) / 2 - human_center_point[0]
                      y_cord = (role_bbox[1,1] + role_bbox[1,3]) / 2 - human_center_point[1]
                  
                
                y.append((x_cord, y_cord))
                human_center.append(human_center_point)

        return np.array(X), np.array(y), np.array(original_image), np.array(human_center)