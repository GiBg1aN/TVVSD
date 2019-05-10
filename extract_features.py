import glob
import os
import re

import imghdr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch

from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
from visual_verb_disambiguation import filter_image_name


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


class Identity(nn.Module):
    """Simple identity NN neuron"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# REF: https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
def image_preprocessing(path_img):
    """
    Preprocess 'input_img' by scaling and normalisation, converting it
    to a Pytorch tensor.

    Args:
        path_img: input jpeg image

    Returns:
        A scaled and normalised version of 'input_img' stored in a
        Pytorch tensor
    """
    if imghdr.what(path_img) is None:
        raise ValueError('Invalid image')
    input_img = Image.open(path_img).convert('RGB')

    # Now that we have an img, we need to preprocess it.
    # We need to:
    #       * resize the img, it is pretty big (~1200x1200px).
    #       * normalize it, as noted in the PyTorch pretrained models doc,
    #         with, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    #       * convert it to a PyTorch Tensor.
    #
    # We can do all this preprocessing using a transform pipeline.
    min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(input_img)
    # plt.imshow(img.permute(1,2,0))
    # plt.show()


    # PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels,
    # height, width).
    #
    # Currently however, we have (num color channels, height, width); let's fix this by inserting
    # a new axis.
    img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.

    # Now that we have preprocessed our img, we need to convert it into a
    # Variable; PyTorch models expect inputs to be Variables. A PyTorch Variable is a
    # wrapper around a PyTorch Tensor.
    img = Variable(img)


    return img


def encode_verse_images(vgg16):
    images_df = pd.DataFrame(columns=['e_image'])
    captions_sense_labels = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv')
    captions_sense_labels = captions_sense_labels[['image', 'COCO/TUHOI']].drop_duplicates()
    for _, row in tqdm(enumerate(captions_sense_labels.itertuples())):
        dataset_folder = row[-1]
        path_img = 'data/images/' + dataset_folder + '/' + row.image
        tensor_img = image_preprocessing(path_img).cuda()
        prediction = vgg16(tensor_img)  # Returns a Tensor of shape (batch, num class labels)
        encoded_tensor = prediction.data.cpu().numpy()
        images_df = images_df.append(pd.Series({'e_image': encoded_tensor}, name=row.image),
                                     ignore_index=False)

    captions_sense_labels['image'] = captions_sense_labels['image'].apply(filter_image_name)
    images_df.to_pickle('generated/images_features.pkl')


def encode_verse_senses(vgg16, device):
    queries = pd.read_csv('data/labels/sense_specific_search_engine_queries.tsv', sep='\t', dtype={'sense_num': str})
    queries['sense_num'] = queries['sense_num'].apply(lambda x: x.replace('.', '_'))
    queries['queries'] = queries['queries'].apply(lambda s: s.split(','))

    DOWNLOAD_FOLDER = 'bing_download_azure/'
    enconding_folder = 'sense_features/'

    for i, row in tqdm(enumerate(queries.itertuples())):
        verb = getattr(row, 'verb')
        sense_num = getattr(row, 'sense_num')
        sense_directory = DOWNLOAD_FOLDER + verb + '_' + sense_num + '/'

        queries_directories = glob.glob(sense_directory + '/*')
        queries_directories.sort(key=natural_sort_key)
        features_list = []
        image_names_list = []

        for query_directory in queries_directories:
            vectors = None
            files = glob.glob(query_directory + '/*')
            files.sort(key=natural_sort_key)  # Sort by-name to take the first n images

            for image_path in files:
                try:
                    tensor_img = image_preprocessing(image_path).to(device)
                    image_names_list.append(image_path.split('/')[-1])
                    prediction = vgg16(tensor_img)  # Returns a Tensor of shape (batch, num class labels)
                    encoded_tensor = prediction.data

                    if vectors is None:
                        vectors = encoded_tensor
                    else:
                        vectors = torch.cat([vectors, encoded_tensor])
                except ValueError:
                    pass
                    #print("invalid image: %s" % image_path)
                except OSError:
                    pass
                    #print("invalid image: %s" % image_path)
                except AttributeError:
                    pass
                    #print("invalid image: %s" % image_path)
                    
                    
            if vectors is not None:
                features_list.append(vectors.cpu().numpy())

        filepath = enconding_folder + verb + '_' + sense_num + '.npz'
        print('Writing %s...' % filepath)
        np.savez(filepath, features=features_list, names=image_names_list)


def main():
    cuda = torch.device('cuda:1')
    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(cuda)
    class_layers = (list(vgg16.classifier.children())[:-1])
    class_layers.append(Identity())
    vgg16.classifier = nn.Sequential(*class_layers)

    # encode_verse_images(vgg16)
    encode_verse_senses(vgg16, cuda)
 

if __name__ == '__main__':
    main()
