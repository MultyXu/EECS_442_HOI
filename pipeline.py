'''

1. yolo8: input image -> croped human, object image 
xtract_human_object(image)

2. feature: ectrac

3. 

'''
from ultralytics import YOLO
from ResNet50_extractor import *
from MLP_POI import MLP_POI
import cv2
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm
from PIL import Image


import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_tensor


def extract_human_object(image):
    '''
    input: image
    output: 
    cropped_human, cropped_object (list of cropped image of human and object)
    object_label (list of object id/name)
    '''
    cropped_human = []
    cropped_objects = []
    label_objects = []
    
    model = YOLO('yolov8n.pt')

    results = model(image)

    boxes = results[0].boxes.xyxy.tolist()
    class_ids = results[0].boxes.cls.tolist()

    class_names = model.names

    original_size = image.size

    to_tensor = TOTensor()

    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[class_id]

        canvas = torch.zeros_like(to_tensor(image))
        
        selected = to_tensor(image)[:, y1:y2, x1:x2]
        canvas[:, y1:y2, x1:x2] = selected
      
        if label.lower() == 'person':
            cropped_human.append(canvas)
        else:
            cropped_objects.append(canvas)
            label_objects.append(class_id)

    return cropped_human, cropped_objects, label_objects

def extract_feature(cropped_image):
    '''
    input: a cropped image
    output: features (torch.tensor of image features)
    '''
    feature_extractor = Feature_extractor()
    return feature_extractor.generate_feature(cropped_image)


def train_model(model, dataloaders, criterion, optimizer, save_dir = None, num_epochs=25, model_name='MiniVGG'):
    """
    Args:
        model: The NN to train
        dataloaders: A dictionary containing at least the keys
                    'train','val' that maps to Pytorch data loaders for the dataset
        criterion: The Loss function
        optimizer: Pytroch optimizer. The algorithm to update weights
        num_epochs: How many epochs to train for
        save_dir: Where to save the best model weights that are found. Using None will not write anything to disk.

    Returns:
        model: The trained NN
        tr_acc_history: list, training accuracy history. Recording freq: one epoch.
        val_acc_history: list, validation accuracy history. Recording freq: one epoch.
    """

    val_acc_history = []
    tr_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # loss and number of correct prediction for the current batch
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # TQDM has nice progress bars
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                ###############################################################
                # TODO:                                                       #
                # Please read all the inputs carefully!                       #
                # For "train" phase:                                          #
                # (i)   Compute the outputs using the model                   #
                #       Also, use the  outputs to calculate the class         #
                #       predicted by the model,                               #
                #       Store the predicted class in 'preds'                  #
                #       (Think: argmax of outputs across a dimension)         #
                #       torch.max() might help!                               #
                # (ii)  Use criterion to store the loss in 'loss'             #
                # (iii) Update the model parameters                           #
                # Notes:                                                      #
                # - Don't forget to zero the gradients before beginning the   #
                # loop!                                                       #
                # - "val" phase is the same as train, but without backprop    #
                # - Compute the outputs (Same as "train", calculate 'preds'   #
                # too),                                                       #
                # - Calculate the loss and store it in 'loss'                 #
                ###############################################################

                optimizer.zero_grad() # zero the grad
                outputs = model.forward(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels.data)
                if phase == 'train':
                  loss.backward()
                  optimizer.step()


                ###############################################################
                #                         END OF YOUR CODE                    #
                ###############################################################

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # save the best model weights
                # =========================================================== #
                # IMPORTANT:
                # Losing your connection to colab will lead to loss of trained
                # weights.
                # You should download the trained weights to your local machine.
                # Later, you can load these weights directly without needing to
                # train the neural networks again.
                # =========================================================== #
                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir, model_name + '.pth'))

            # record the train/val accuracies
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                tr_acc_history.append(epoch_acc)

    print('Best val Acc: {:4f}'.format(best_acc))

    return model, tr_acc_history, val_acc_history

if __name__=='__main__':
    print("Excuting")
    
    input_size = 4096 #depends on the feature it will be fed into
    layers = [2048, 512, 128] # define number of layer and layer size, need to experiment
    LR = 1e-2 # learning rate
    DECAY = 1e-2 # decay rate
    
    interactive_net = MLP_POI(input_size, layers)
    optimizer = optim.Adam(interactive_net.parameters, lr=LR, weight_decay=DECAY)
    loss = nn.MSELoss()