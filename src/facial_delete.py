import pathlib
from tqdm import tqdm


data_imgs = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data/finalfitz17k")
facial_data_imgs = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data/facial")

count = 0
print("starting...")
for facial_img in tqdm(sorted(facial_data_imgs.iterdir())):
    main_img = data_imgs / facial_img.name 

    if main_img.exists():
        main_img.unlink()
        count += 1
    else:
       continue

print(count)

