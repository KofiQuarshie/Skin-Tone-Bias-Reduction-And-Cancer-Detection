# Skin-Tone-Bias-Reduction-And-Cancer-Detection
This project is implements a cancer detection ML model but reduced skin tone bias for type V-VI skin types. Various augmentations, fine-tuning, synthetic image generations are used to help reduce the gap in detecting cancer in darker skin images compared to lighter skinned images

## ACTIVATE VIRTUAL ENVIRONEMENT
- python3.15 -m venv .venv
- source .venv/bin/activate
- pip install pandas numpy pillow matplotlib scikit-learn


## RUNNING YUNET FACIAL RECOGNITION MODEL
This feature is being implemented to resolve the ethical concerns that are related to identifiable facial images in the dataset, specifically fitzpatrick17k. could be used on others too
- Install csv, shutil, cv2 (optional tqdm to load progress bar)
- Point data path to dataset to recognise dataset images
- New csv is created for flagged facial images for reference
- PAD_UFES and DDI had no identifiable images recorded, Fitzpatrick had 633 images, CSV can be found at facial_csv/facial.csv

## REMOVING FACIAL IMAGES FROM FITZPATRICK17K DATASET
This feature should only be implemented after Yunet facial detection has been run
- Run facial_delete.py to remove all images with faces permanently from the main dataset

## CLEANING FITZPATRICK17K DATASET
fitzpatrick_clean.py file is used to further clean the dataset after the removal of identifiable facial images. 
- Firstly, it checks if every image in the fitzpatrick17k folder is present in the csv file, if not, the missing ones (facial images) are dropped from th csv file
- Secondly, fitzpatrick_scale row determines the skin group of patients in the dataset, 1-6 varying from lightest to darkest. A scale of -1 means it cannot be identified and this is also dropped from the csv file
- Also, we only want skin diseases which are either malignant or benign, not non-neoplastic(not cancer nor a grpwth, could be eczema, acne, fungal infections, etc), which is outside the scope of this project. Non-neoplastic labels rows are also dropped
- Lastly, the quality control (qc) column have been reviewed by dermatologists and given their remarks, we only keep characteristic or diagnostic values which shows there may be a growth of cancer or there definetely is cancer respectively. NaN values are also used, mainly because they havent been reviewed by experts yet. Other values(possiblu, other, wrongly labelled) will not be used to train the model and are dropped.

