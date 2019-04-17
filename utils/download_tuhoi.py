import pandas as pd
import requests
import re

from tqdm import tqdm


sense_labels = pd.read_csv('full_sense_annotations.csv')
sense_labels = sense_labels[sense_labels['COCO/TUHOI'] == 'TUHOI']

url = 'http://169.44.201.108:7002/imagenet/'
download_folder = '/run/media/gibg1an/Dati/Download/dataset/'

image_2013_count = 0

for i, row in tqdm(enumerate(sense_labels.itertuples())):
    if row.image.startswith('ILSVRC2012_val'):
        folder = 'val/'
    elif row.image.startswith('n'):
        folder = re.match("^n\d+", row.image).group()
    elif row.image.startswith('ILSVRC2013_val'):
        image_2013_count += 1
        continue
    else:
        print('Unknown prefix: %s' % row.image)

    r = requests.get(url + folder + row.image)

    with open(download_folder + row.image, 'wb') as f:
        f.write(r.content)

print('%s ILSVRC2013 items' % image_2013_count)
