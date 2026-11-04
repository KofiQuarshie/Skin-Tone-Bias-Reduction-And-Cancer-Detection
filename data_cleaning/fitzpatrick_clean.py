import csv
import pandas as pd
import pathlib


data_folder = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/Fitzpatrick/data")
data_imgs = data_folder / 'finalfitz17k'

imgs = set()
df = pd.read_csv(data_folder/'fitzpatrick17k.csv')

for img in sorted(data_imgs.iterdir()):
    imgs.add(img.stem)

df = df[df["md5hash"].isin(imgs)]       ##goes through row and Trie if md5hash is in image, keeps  True rows
print(f"Rows with a matching image: {len(df)}")

df = df[df["fitzpatrick_scale"] != -1]      ##-1 if skin type is not known in fitzpatrick, keep ones which are known
print(f"Rows with a known skin type: {len(df)}")

remove_partition_label = ["malignant", "benign"]
df = df[df["three_partition_label"].isin(remove_partition_label)]
print(f"Rows which are cancer or benign: {len(df)}")

remove_qc = ["5 Potentially", "3 Wrongly labelled", "4 Other"]
df = df[~df["qc"].isin(remove_qc)]      ##keep if qc is potentially or wronly lableed, ~ negates it so keep if False
print(f"Rows with a known or charateristic cancer: {len(df)}")


disease_dict = {"malignant": 1, "benign": 0}
#label_dict.map(label)
#label = [0, 1]
skin_group_dict = {1: "light", 2: "light", 3: "medium", 4: "medium", 5: "dark", 6: "dark"}

df['disease'] = df["three_partition_label"].map(disease_dict)
df['skin_group'] = df["fitzpatrick_scale"].map(skin_group_dict)

print(df["disease"].value_counts(dropna=False))
print(df["skin_group"].value_counts(dropna=False))


df.to_csv(data_folder/'cleaned_fitzpatrick.csv', index = False)

##checking the ratio of skin groups and their disease(whether they have cancer or not)
res = pd.crosstab(df['disease'], df['skin_group'])
print(res)
#print(csv_read.head())