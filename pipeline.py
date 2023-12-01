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
from torchvision.transforms import ToTensor

def extract_human_object(image):
    '''
    input: image (a torch tensor, 3x640x640, 0~255)
    output: 
    cropped_human, cropped_object (list of cropped image of human and object)
    id_obejects (list of object id/name)
    '''
    # Initialize output values:
    cropped_human = []
    cropped_objects = []
    human_box = []
    object_box = []
    id_objects = []

    # change image_tensor size to be yolo fit
    yolo_img = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    print("image_shape", image.shape)
    
    # Load Yolo model for object detection
    model_objects = YOLO('yolov8n.pt')
    model_human = YOLO('yolov8n-pose.pt')
    
    results_objects = model_objects(yolo_img)
    results_hunan = model_human(yolo_img)

    # Bounding boxes of detected objects
    boxes_objects = results_objects[0].boxes.xyxy.tolist()
    boxes_human = results_hunan[0].boxes.xyxy.tolist()
    
    # Class IDs of detected objects
    class_ids_objects = results_objects[0].boxes.cls.tolist()
    class_ids_human = results_hunan[0].boxes.cls.tolist()

    # The names of classes that the YOLO model can detect
    class_names_objects = model_objects.names
    class_names_human = model_human.names

    to_tensor = ToTensor()

    for _, (box, class_id) in enumerate(zip(boxes_objects, class_ids_objects)):
        # Bounding box & label for each object
        x1, y1, x2, y2 = map(int, box)
        label = class_names_objects[class_id]

        # A black image of the same size as the input image
        canvas = torch.zeros_like(image)
        selected = image[:, y1:y2, x1:x2]
        # Crop and reprint the selected image
        canvas[:, y1:y2, x1:x2] = selected

        # Add to return value - cropped_objects & id_objects
        if label.lower() != 'person':
            cropped_objects.append(canvas)
            id_objects.append(class_id)
            object_box.append([x1, y1, x2, y2])

    for _, (box, class_id) in enumerate(zip(boxes_human, class_ids_human)):
        # Bouding box &label for each object
        x1, y1, x2, y2 = map(int, box)
        label = class_names_human[class_id]

        # A black image of the same size as the input image
        canvas = torch.zeros_like(image)
        selected = image[:, y1:y2, x1:x2]
        # Crop and reprint the selected image
        canvas[:, y1:y2, x1:x2] = selected

        # Add to return value - cropped_human
        if label.lower() == 'person':
            cropped_human.append(canvas)
            human_box.append([x1, y1, x2, y2])

    return cropped_human, cropped_objects, human_box, object_box, id_objects

def extract_feature(cropped_image, device='cpu'):
    '''
    input: a cropped image
    output: features (torch.tensor of image features)
    '''
    feature_extractor = Feature_extractor(device)
    return feature_extractor.generate_feature(cropped_image)


def inference(image_tensor, mlp_model):
  '''
  input: 
    image_tensor (3x640x640, 0~255)
    mlp_model (traind mlp classifier)
  output: 
    list of point of interest (x,y)
    list of human box (x,y,x,y)
    list of object box (x,y,x,y)
    list of id_objects (id)
  '''
  # store list of poi (point of interest) corresponding to each person 
  poi = []

  cropped_human, cropped_objects, human_box, object_box, id_objects = extract_human_object(image_tensor / 255)

  for human in cropped_human:
    feature = extract_feature(human)['layer4']
    result_poi = mlp_model.forward(feature)
    poi.append(result_poi)
  
  return result_poi, human_box, object_box, id_objects



def train_model(model, dataloaders, criterion, optimizer, device='cpu', vis=False, save_dir = None, num_epochs=25, model_name='MLP_POI'):
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
        for phase in ['train', 'test']:
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

                # extract the feature of cropped image to layer4 dimension
                features = extract_feature(inputs, device)['layer4']

                # predict the point of interest
                outputs = model(features)

                # compute loss
                # preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                # print(inputs[0,0,0,0])
                # print(type(loss))
                # print(type(labels))
                if phase == 'train':
                  loss.backward()
                  optimizer.step()


                ###############################################################
                #                         END OF YOUR CODE                    #
                ###############################################################

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # hard code correct constraint 
                # if difference of l2 norm between GT poi and pred poi < 50
                diff = torch.linalg.norm((outputs - labels)*640, ord=2, dim=1)
                correct_pred = torch.where(diff < 50, 1, 0)
                running_corrects += torch.count_nonzero(correct_pred)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # visualize the prediction result 
                if vis:
                  for i in len(inputs):
                    plt.figure()
                    plt.imshow[inputs[i].permute(1,2,0)]
                    plt.plot(labels[0][0], labels[0][1], marker='.', markersize=20)


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
            if phase == 'test':
                val_acc_history.append(epoch_acc)
            else:
                tr_acc_history.append(epoch_acc)

    print('Best test Acc: {:4f}'.format(best_acc))

    return model, tr_acc_history, val_acc_history

def main():
    print("Excuting")
    
    input_size = 4096 #depends on the feature it will be fed into
    layers = [2048, 512, 128] # define number of layer and layer size, need to experiment
    LR = 1e-2 # learning rate
    DECAY = 1e-2 # decay rate
    
    interactive_net = MLP_POI(input_size, layers)
    optimizer = optim.Adam(interactive_net.parameters, lr=LR, weight_decay=DECAY)
    loss = nn.MSELoss()