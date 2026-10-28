import pathlib
import numpy as np
import cv2
from tqdm import tqdm       #load progress bar
import shutil
import csv


data_imgs = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data/finalfitz17k")
facial_data_imgs = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data/facial")
facial_data_imgs.mkdir(parents=True, exist_ok=True)

csv_results = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data/facial_csv")
csv_results.mkdir(parents=True, exist_ok=True)

model = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/models/face_detection_yunet_2023mar.onnx")


yunet_model = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_threshold=0.8)    #using Yunet face detector to remove facial images in dataset


count = 0
unreadable_imgs = []
flagged_imgs = []

print("starting...")
for img in tqdm(sorted(data_imgs.iterdir())):
    if img.suffix == ".jpg":
        count += 1
        image = cv2.imread(str(img))

        if image is None:
            unreadable_imgs.append(img.name)
            continue
    
    
    height, width = image.shape[:2]     #facial array size(height,width,channels), slicing channels off
    yunet_model.setInputSize((width, height))
    _, faces = yunet_model.detect(image)    #status value - ignored, face stored in variable

        if faces is not None:
            num_faces = len(faces)      #number of faces in an image
            max_score = float(faces[:, -1].max())       #stores maximum face detection confidence

            face_areas = faces[:, 2] * faces[:, 3]
            largest_face_area = float(face_areas.max())
            face_area_ratio = largest_face_area / (width * height)      #how much of the picture the face consumes

            #storing flagged facial images in a dictionary
            flagged_imgs.append({
                "filename": img.name,
                "num_faces": num_faces,
                "max_score": round(max_score, 3),
                "face_area_ratio": round(face_area_ratio, 4),
            })

            #if flagged_imgs in data_imgs:
            shutil.copy(img, facial_data_imgs)
len_flagged = len(flagged_imgs)
                
#save new csv file
with open(csv_results/'facial.csv','w', newline = '') as f:
    w = csv.DictWriter(f,fieldnames=["filename", "num_faces", "max_score", "face_area_ratio"])
    w.writeheader()
    w.writerows(flagged_imgs)


print(f"{len_flagged}")




