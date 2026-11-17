import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0(nn.Module):

    def __init__(self, nch_out=1):
        super(EfficientNetB0, self).__init__()


        self.model = efficientnet_b0(
            weights = EfficientNet_B0_Weights.DEFAULT,  #load pre-trained weights so model has visual kmowledge of images before training
            #output only 1 channel (nch_out), 1 = cancer probability
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, nch_out)
            progress = True
        )

    def forward(self, x):
        return self.model(x)
    




