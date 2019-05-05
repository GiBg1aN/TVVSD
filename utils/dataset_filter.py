"""
Group single-verb VerSe images to a common folder
"""
from shutil import copyfile
import pandas as pd

def dataset_filter():
    """
    Copy images listed in 'full_sense_annotations.csv' to 'dataset'
    directory
    """
    sense_labels_df = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv')
    sense_labels_df = sense_labels_df[sense_labels_df['COCO/TUHOI'] == 'COCO']
    sense_labels_df = sense_labels_df['image'].drop_duplicates()

    for filename in sense_labels_df:
        if filename.startswith('COCO_train2014'):
            copyfile('zips/train2014/' + filename, 'dataset/' + filename)
        if filename.startswith('COCO_val2014'):
            copyfile('zips/val2014/' + filename, 'dataset/' + filename)


if __name__ == '__main__':
    dataset_filter()
