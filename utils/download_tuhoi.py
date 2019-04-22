"""
Single-verb VerSe images are downloaded from TUHOI repository.
"""
import pandas as pd
import requests
from tqdm import tqdm


SENSE_LABELS = pd.read_csv('data/labels/full_sense_annotations_filtered.csv')
SENSE_LABELS = SENSE_LABELS[SENSE_LABELS['COCO/TUHOI'] == 'TUHOI']

URL = 'http://disi.unitn.it/~dle/images/DET/'
DOWNLOAD_FOLDER = 'data/images/TUHOI/'

for i, row in tqdm(enumerate(SENSE_LABELS.itertuples())):
    if row.image.startswith('ILSVRC'):
        folder = 'val/'
    elif row.image.startswith('n'):
        folder = 'train/'
    else:
        print('Unknown prefix, image name: %s' % row.image)

    r = requests.get(URL + folder + row.image)

    with open(DOWNLOAD_FOLDER + row.image, 'wb') as f:
        f.write(r.content)
