"""
datasets
"""

import os
import random

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import yaml


class Fitzpatrick(Dataset):
    

    def __init__(self, csv_path, f_dir, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.f_dir = f_dir
        
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.f_dir, row["md5hash"] + ".jpg")
        image = read_image(str(img_path)).type(torch.float) / 255
        if image.shape[0] == 1: # grayscale -> RGB
            image = image.repeat(3, 1, 1)
    
        if self.transform:
            image = self.transform(image)

        return image

data_folder = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data/finalfitz17k")
with open ("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/configs/config_fitzpatrick.yaml",'r', newline = '') as f:
    config = yaml.safe_load(f)

train_set = Fitzpatrick(csv_path=config["datasets_splitting_ratios"]["train_dataset_csv"], f_dir=data_folder, transform=None)
test_set  = Fitzpatrick(csv_path=config["datasets_splitting_ratios"]["test_dataset_csv"],  f_dir=data_folder, transform=None)
#val_set = Fitzpatrick(csv_path=config["datasets_splitting_ratios"]["val_dataset_csv"],  f_dir=data_folder, transform=test_transforms)

print(len(train_set))
print(train_set[0])
print(train_set[0].shape)