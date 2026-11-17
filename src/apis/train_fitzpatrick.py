
from pathlib import Path
import torch
import torchvision.transforms.v2 as transforms 
from argparse import ArgumentParser


from src.utils.dataset import Fitzpatrick
import yaml

def main(args):

    
    transforms_img = transforms.Compose([
                                            #transforms.ToImage(),
                                            transforms.Resize((224,224)),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(20),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ##efficientnet model Normalisation pre-trained values
                                            ])

    transforms_rotations = transforms.Compose([
                                             transforms.Resize((224,224)),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_map = {
    'transforms_img': transforms_img,
    'transforms_rotations': transforms_rotations
    }

    with open (args.config_file,'r', newline = '') as f:
        config = yaml.safe_load(f)

    train_transform = transform_map.get(config["transforms"]["train_transform_operation"])
    test_transform = transform_map.get(config["transforms"]["test_transform_operation"])
    
        
    train_set = Fitzpatrick(csv_path=config["datasets_splitting_ratios"]["train_dataset_csv"], f_dir=config["dataset"]["img_path"], transform=train_transform)
    test_set  = Fitzpatrick(csv_path=config["datasets_splitting_ratios"]["test_dataset_csv"],  f_dir=config["dataset"]["img_path"], transform=test_transform)
    val_set = Fitzpatrick(csv_path=config["datasets_splitting_ratios"]["val_dataset_csv"],  f_dir=config["dataset"]["img_path"],  transform=test_transform)

    print("train set: ", len(train_set))
    print("test set: ", len(test_set))
    print("val set: ", len(val_set))

    
    print(train_set[0][0].shape)
    print(train_set[0][1])
    print(train_set[0][2])
if __name__ == "__main__":

    parser = ArgumentParser(description="Skin Tone Bias Reduction")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)

    args = parser.parse_args()
    main(args)