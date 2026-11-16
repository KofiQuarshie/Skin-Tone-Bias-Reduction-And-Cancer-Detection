import csv
import pandas as pd
import pathlib


data_folder = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/PAD_UFES/zr7vgbcyr2-1")
data_imgs1 = data_folder / 'images/imgs_part_1'
data_imgs2 = data_folder / 'images/imgs_part_2'
data_imgs3 = data_folder / 'images/imgs_part_3'

imgs = set()
df = pd.read_csv(data_folder/'metadata.csv')

#print(df.head())
for img in sorted(data_imgs1.iterdir()):
    #image = cv2.imread(str(img))
    imgs.add(img.name)

for img in sorted(data_imgs2.iterdir()):
    imgs.add(img.name)

for img in sorted(data_imgs3.iterdir()):
    imgs.add(img.name)

df = df[df["img_id"].isin(imgs)]       
print(f"Rows with a matching image: {len(df)}")

print(df["diagnostic"].value_counts(dropna=False))
diagnostic_values = ["BCC", "SCC", "MEL", "NEV", "SEK"]
df = df[df["diagnostic"].isin(diagnostic_values)]
print(f"Rows which are cancer or benign: {len(df)}")

print(df["fitspatrick"].value_counts(dropna=False))

disease_dict = {"BCC": 1,"SCC": 1, "MEL": 1, "NEV": 0, "SEK": 0}
skin_group_dict = {1: "light", 2: "light", 3: "medium", 4: "medium", 5: "dark", 6: "dark"}

df['disease'] = df["diagnostic"].map(disease_dict)
df['skin_group'] = df["fitspatrick"].map(skin_group_dict)

print(df["disease"].value_counts(dropna=False))
print(df["skin_group"].value_counts(dropna=False))


df.to_csv(data_folder/'cleaned_pad_ufes.csv', index = False)

##checking the ratio of skin groups and their disease(whether they have cancer or not)
res = pd.crosstab(df['disease'], df['skin_group'])
print(res)
