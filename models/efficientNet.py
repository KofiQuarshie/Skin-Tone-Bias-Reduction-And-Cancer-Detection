import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt


class EfficientNetB0(nn.Module):

    def __init__(self, nch_out=1):
        super(EfficientNetB0, self).__init__()


        self.model = EfficientNet(
            weights = None,  #No pretrained weights so model can learn rather than memorise patterns
            weights_backbone="DEFAULT",

        )
    

    #def efficientnet_b0(
    #*, weights: Optional[EfficientNet_B0_Weights] = None, progress: bool = True, **kwargs: Any)
    #weights = EfficientNet_B0_Weights.verify(weights)

    #inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    #return _efficientnet(
     #   inverted_residual_setting, kwargs.pop("dropout", 0.2), last_channel, weights, progress, **kwargs
    #)

    def create_b0_model():
        b0_model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights='imagenet', 
        input_shape=([3, 224, 224]),
        pooling='avg'
    )

    ## Freeze only the bottom layers of the model
    tuned = int(len(b0_model.layers) *.85)
    for layer in b0_model.layers[:tuned]:
        layer.trainable = False

    model = tf.keras.Sequential([
        b0_model,
        ## Add dropout layer with .3 dropout
        tf.keras.layers.Dropout(0.4),
        ## Add dense layer
        tf.keras.layers.Dense(256, activation='relu'),
        ## Add another dropout layer with .2 dropout
        tf.keras.layers.Dropout(0.3),
        ## This is the output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #model = create_b0_model()
    
    return model





