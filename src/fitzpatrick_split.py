import csv
import pandas as pd
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

data_folder = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data")
df = pd.read_csv(data_folder/'cleaned_fitzpatrick.csv')
print(df.shape)

df['combined_cols'] = df['disease'].astype(str) + "_" + df['skin_group']
print(df['combined_cols'].value_counts())


with open ("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/configs/config_fitzpatrick.yaml",'r', newline = '') as f:
    config = yaml.safe_load(f)

# using the train test split function
train_val_set, test_set = train_test_split(
  df, random_state=config["datasets_splitting_ratios"]["seed"] ,test_size =config["datasets_splitting_ratios"]["test_set"], stratify = df['combined_cols'], shuffle=True)
## 0.176 because we want 15% for validation of the entire data as 15% was initially used for testing  so 0.15 / train set(70) + val_set(0.15) left = 
train_set, val_set = train_test_split(
    train_val_set, random_state=config["datasets_splitting_ratios"]["seed"], test_size = 0.176, stratify = train_val_set['combined_cols']
)

print('')
print('X_test : ')
print(test_set.head())
print(test_set.shape)

print('')
print('y_train : ')
print(train_set.head())
print(train_set.shape)

print('')
print('val_set : ')
print(val_set.head())
print(val_set.shape)

train_set.to_csv(data_folder/'train_split.csv', index = False)
test_set.to_csv(data_folder/'test_split.csv', index = False)
val_set.to_csv(data_folder/'val_split.csv', index = False)