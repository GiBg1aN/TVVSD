from shutil import copyfile

import pandas as pd

def dataset_filter():
    """
    Copy images listed in 'full_sense_annotations.csv' to 'dataset'
    directory
    """
    sense_labels_df = pd.read_csv('full_sense_annotations.csv')
    sense_labels_df = sense_labels_df[sense_labels_df['COCO/TUHOI'] == 'COCO']

    for filename in sense_labels_df['image']:
        if filename.startswith('COCO_train2014'):
            copyfile('train2014/' + filename, 'dataset/' + filename)
        if filename.startswith('COCO_val2014'):
            copyfile('val2014/' + filename, 'dataset/' + filename)
