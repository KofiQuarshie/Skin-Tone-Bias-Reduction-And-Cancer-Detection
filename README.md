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