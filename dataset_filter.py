import os
import pandas as pd

from shutil import copyfile


df = pd.read_csv("full_sense_annotations.csv")

df = df[df['COCO/TUHOI'] == 'COCO']

for filename in df['image']:
    if filename.startswith("COCO_train2014"):
        copyfile("train2014/" + filename, "dataset/" + filename)
    if filename.startswith("COCO_val2014"):
        copyfile("val2014/" + filename, "dataset/" + filename)
