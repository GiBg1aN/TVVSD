import numpy as np
import pandas as pd


def filter_image_name(img_name):
    train_prefix = "COCO_train2014_"
    val_prefix = "COCO_val2014_"
    if img_name.startswith(train_prefix):
        return int(img_name[len(train_prefix):-4])
    if img_name.startswith(val_prefix):
        return int(img_name[len(val_prefix):-4])
    raise ValueError('image prefix nor train and val')


def simple_disambiguation(images, senses, labels):
    accuracy = [0, 0]
    for i in range(len(images)):
        i_t = np.array(images.iloc[i]['caption'])
        image_id = images.iloc[i].name
        verbs = labels.query('image == @image_id')["lemma"]

        for j in range(len(verbs)):
            verb = verbs.iloc[j]
            filtered_senses = senses.query("lemma == @verb")
            dot_product = filtered_senses['definition'].apply(lambda s_t: np.dot(i_t, s_t)).to_numpy()
            s_hat = np.argmax(dot_product)
            pred_sense_id = filtered_senses.iloc[s_hat]['sense_num']
            sense_id = labels.query("image == @image_id and lemma == @verb")['sense_chosen'].iloc[0]

            if sense_id == pred_sense_id:
                accuracy[1] += 1
            else:
                accuracy[0] += 1

    accuracy = (accuracy[1] / (accuracy[0] + accuracy[1])) * 100

    print("Sense accuracy is: %s" % accuracy)
        

def main():
    embedded_captions = pd.read_pickle("embedded_captions.pkl")
    embedded_senses = pd.read_pickle("embedded_senses.pkl")
    captions_sense_labels = pd.read_csv("full_sense_annotations.csv")
    captions_sense_labels = captions_sense_labels[captions_sense_labels['COCO/TUHOI'] == 'COCO']
    captions_sense_labels['image'] = captions_sense_labels['image'].apply(filter_image_name)

    simple_disambiguation(embedded_captions, embedded_senses, captions_sense_labels)


if __name__ == '__main__':
    main()


