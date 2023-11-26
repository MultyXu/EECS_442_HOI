import torch
import torchvision
import torch.nn as nn
    
class MLP_POI(nn.Module):

    def __init__(self, input_size, layers, drop_out_rate=0.3, num_classes=2, init_weights=True):
        super(MLP_POI, self).__init__()

        # construct a simple MLP that do binary classification on interactiveness
        self.input_size = input_size
        self.layers = layers
        # self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        layer_dic = []
        last_layer = input_size
        for layer in layers:
          layer_dic.append(nn.Linear(last_layer, layer, True))
          layer_dic.append(nn.ReLU(True))
          layer_dic.append(nn.Dropout(0.3))
          last_layer = layer

        layer_dic.append(nn.Linear(last_layer, 2, True))

        self.classifier = nn.Sequential(*layer_dic)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)