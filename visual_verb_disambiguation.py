"""
Perform Visual Verb Sense Disambiguation according to Gella at. al.
method.
"""
import numpy as np
import pandas as pd


def filter_image_name(img_name):
    """
    Remove image name prefixes.

    Args:
        img_name: image name in the form PREFIX_XXXX.jpeg

    Returns:
        The XXXX image identifier

    Raises:
        ValueError: when the image prefix is not known
    """
    train_prefix = 'COCO_train2014_'
    val_prefix = 'COCO_val2014_'
    if img_name.startswith(train_prefix):
        stripped_zeros = train_prefix + str(int(img_name[len(train_prefix):-4]))
    elif img_name.startswith(val_prefix):
        stripped_zeros = val_prefix + str(int(img_name[len(val_prefix):-4]))
    else:
        stripped_zeros = img_name
    return stripped_zeros.split('.')[0]


def simple_disambiguation(images, senses, labels, image_column, verb_types):
    """
    Compute cosine similarity between images and senses representation
    vectors. Accuracy is computed and printed.

    Args:
        images: A dataframe of image representations
        senses: A dataframe of senses representations
        sense_labels: A dataframe that contains the verb and the correct
            sense for each image
        image_column: Column index of 'images'
        verb_types: Dictionary split in 'motion' and 'non-motion' verbs

    Returns:
        None

    Raises:
        ValueError: when the current verb is not in motion verbs list,
            nor in non-motion verbs list.
    """
    accuracy = {'motion': [0, 0], 'non_motion': [0, 0]}
    for _, image_row in enumerate(images.itertuples()):
        i_t = np.array(getattr(image_row, image_column))
        image_id = image_row.Index
        verbs = labels.query('image == @image_id')['lemma'].to_frame()

        for _, verb_row in enumerate(verbs.itertuples()):
            verb = verb_row.lemma
            filtered_senses = senses.query('lemma == @verb')
            # Cosine similarity between image i_t and every other sense s_t
            dot_prod = filtered_senses['e_combined'].apply(
                lambda s_t: -1 if np.all(i_t == None) else np.dot(i_t, s_t))
            s_hat = np.argmax(dot_prod)
            if np.max(dot_prod) == -1:  # the image can't be represented
                continue
            pred_sense_id = filtered_senses.iloc[s_hat]['sense_num']
            sense_id = labels.query('image == @image_id and lemma == @verb')['sense_chosen'].iloc[0]
            if sense_id == -1:  # -1 sense_id do not exists in OntoNotes
                continue

            # Accuracy statistics
            if verb in verb_types['motion']:
                if sense_id == pred_sense_id:
                    accuracy['motion'][1] += 1
                else:
                    accuracy['motion'][0] += 1
            elif verb in verb_types['non_motion']:
                if sense_id == pred_sense_id:
                    accuracy['non_motion'][1] += 1
                else:
                    accuracy['non_motion'][0] += 1
            else:
                raise ValueError('Unknown verb type')

    print('%s representation, sense accuracy:' % image_column)
    print('Motion verbs: %s' % ((accuracy['motion'][1] / (accuracy['motion'][0] + accuracy['motion'][1])) * 100))
    print('Non-motion verbs: %s' % ((accuracy['non_motion'][1] / (accuracy['non_motion'][0] + accuracy['non_motion'][1])) * 100))
    print('-')


def main():
    "Load embedded image and caption, and disambiguate senses."
    embedded_captions = pd.read_pickle('generated/embedded_annotations.pkl')
    embedded_senses = pd.read_pickle('generated/embedded_senses.pkl')
    captions_sense_labels = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv')
    captions_sense_labels['image'] = captions_sense_labels['image'].apply(filter_image_name)

    verb_types = {}

    with open('data/labels/motion_verbs.csv') as motion_verbs:
        verb_types['motion'] = [line.rstrip('\n') for line in motion_verbs]

    with open('data/labels/non_motion_verbs.csv') as non_motion_verbs:
        verb_types['non_motion'] = [line.rstrip('\n') for line in non_motion_verbs]

    for representation_type in embedded_captions.columns.to_list():
        simple_disambiguation(embedded_captions, embedded_senses, captions_sense_labels,
                              representation_type, verb_types)


if __name__ == '__main__':
    main()
