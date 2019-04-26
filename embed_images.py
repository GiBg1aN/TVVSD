import os

import pandas as pd
from PIL import Image
import torch.nn as nn

from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
from visual_verb_disambiguation import filter_image_name

class Identity(nn.Module):
    """Simple identity NN neuron"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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


def main():
    vgg16 = models.vgg16(pretrained=True)
    class_layers = (list(vgg16.classifier.children())[:-1])
    class_layers.append(Identity())
    vgg16.classifier = nn.Sequential(*class_layers)

    images_df = pd.DataFrame(columns=['e_image'])

    captions_sense_labels = pd.read_csv('generated/full_sense_annotations_filtered.csv')

    for _, row in tqdm(enumerate(captions_sense_labels.itertuples())):
        dataset_folder = row[-1]
        path_img = 'data/images/' + dataset_folder + '/' + row.image
        tensor_img = image_preprocessing(path_img)
        prediction = vgg16(tensor_img)  # Returns a Tensor of shape (batch, num class labels)
        encoded_tensor = prediction.data.numpy()
        images_df = images_df.append(pd.Series({'e_image': encoded_tensor}, name=row.image),
                                     ignore_index=False)

    captions_sense_labels['image'] = captions_sense_labels['image'].apply(filter_image_name)
    images_df.to_pickle('generated/embedded_images.pkl')


if __name__ == '__main__':
    main()
