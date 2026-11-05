import csv
import pathlib
import pandas as pd

data_folder = pathlib.Path("/Users/kofi/projects/Skin-Tone-Bias-Reduction-And-Cancer-Detection/data/DDI")
#data_imgs = data_folder

df = pd.read_csv(data_folder/'ddi_metadata.csv')

#print(df.head())
print(df["malignant"].value_counts(dropna=False))
malignant_values = [True, False]
df = df[df["malignant"].isin(malignant_values)]
print(f"Rows which are cancer or benign: {len(df)}")


skin_tone_values = [12, 34, 56]
df = df[df["skin_tone"].isin(skin_tone_values)]    
print(f"Rows with a known skin tone: {len(df)}")


disease_dict = {True: 1, False: 0}
skin_group_dict = {12: "light", 34: "medium",  56: "dark"}

df['disease'] = df["malignant"].map(disease_dict)
df['skin_group'] = df["skin_tone"].map(skin_group_dict)

print(df["disease"].value_counts(dropna=False))
print(df["skin_group"].value_counts(dropna=False))


df.to_csv(data_folder/'cleaned_ddi.csv', index = False)

##checking the ratio of skin groups and their disease(whether they have cancer or not)
res = pd.crosstab(df['disease'], df['skin_group'])
print(res)