import json
import os
import pandas as pd

fcaption_train2014 = open("annotations_trainval2014/annotations/captions_train2014.json")
fcaption_val2014 = open("annotations_trainval2014/annotations/captions_val2014.json")
caption_train = json.load(fcaption_train2014)
caption_val = json.load(fcaption_val2014)

train_annotations_list = caption_train.get("annotations")
val_annotations_list = caption_val.get("annotations")

train_df = pd.DataFrame(train_annotations_list)
val_df = pd.DataFrame(val_annotations_list)

img_list = [os.path.splitext(x)[0] for x in os.listdir("dataset")]


new_df1 = pd.DataFrame(columns=['caption', 'id', 'image_id'])
new_df2 = pd.DataFrame(columns=['caption', 'id', 'image_id'])

for img_name in img_list:
    if img_name.startswith("COCO_train2014_"):
        img_id = int(img_name[len("COCO_train2014_"):])
        new_df1 = pd.concat([new_df1, train_df[train_df['image_id'] == img_id]])
    else:
        img_id = int(img_name[len("COCO_val2014_"):])
        new_df2 = pd.concat([new_df2, val_df[val_df['image_id'] == img_id]])

new_df = pd.concat([new_df1, new_df2])
new_df = new_df.drop(new_df.columns[1], axis=1)

if len(new_df1.image_id.unique()) + len(new_df2.image_id.unique()) == len(new_df.image_id.unique()):
    new_df.to_csv("filtered_annotations.csv")
    print(len(new_df.image_id.unique()), "unique annotations.")
    print("Writing...")
else:
    print("Not unique image IDs, cannot concat dataframes.")


