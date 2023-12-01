from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch

class Feature_extractor():
    def __init__(self, device='cpu'):
        # Step 1: Initialize model with the best available weights
        # weights = ResNet50_Weights.DEFAULT
        weights = ResNet50_Weights.IMAGENET1K_V2#.to(torch.device('cuda:0'))
        model = resnet50(weights=weights).to(device)
        model.eval()

        # Step 2: Initialize the inference transforms
        self.preprocess = weights.transforms()

        # define the node you want to return
        return_nodes = {
            "layer4.2.relu_2": "layer4",
            "layer3.2.relu_2": "layer3"
        }

        self.feature_model = create_feature_extractor(model, return_nodes=return_nodes)

    def generate_feature(self, image):
        '''
        image: a torch tensor returned by torch image read
        given image, produce the feature tensor
        '''
        image_processed = self.preprocess(image)
        feature = self.feature_model(image_processed)
        return feature
