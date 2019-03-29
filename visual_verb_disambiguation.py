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
    verb_accuracy = [0, 0]
    sense_accuracy = [0, 0]
    for i in range(len(images)):
        i_t = np.array(images.iloc[i]['caption'])
        dot_product = senses['definition'].apply(lambda s_t: np.dot(i_t, s_t)).to_numpy()
        s_hat = np.argmax(dot_product)
        pred_lemma = senses.iloc[s_hat]['lemma']
        pred_sense_id = senses.iloc[s_hat]['sense_num']
        lemma = list(pd.Series(labels.loc[images.iloc[i].name]['lemma']))
        sense_id = labels.loc[images.iloc[i].name]['sense_chosen']

        if pred_lemma in lemma:
            verb_accuracy[1] += 1
            if sense_id == pred_sense_id:
                sense_accuracy[1] += 1
            else:
                sense_accuracy[0] += 1
        else:
            verb_accuracy[0] += 1

    verb_accuracy = verb_accuracy[1] / (verb_accuracy[0] + verb_accuracy[1]) * 100
    sense_accuracy = sense_accuracy[1] / (sense_accuracy[0] + sense_accuracy[1]) * 100

    print("Verb accuracy is: %s" % verb_accuracy)
    print("Sense accuracy is: %s" % sense_accuracy)
        

def main():
    embedded_captions = pd.read_pickle("embedded_captions.pkl")
    embedded_senses = pd.read_pickle("embedded_senses.pkl")
    captions_sense_labels = pd.read_csv("full_sense_annotations.csv")
    captions_sense_labels = captions_sense_labels[captions_sense_labels['COCO/TUHOI'] == 'COCO']
    captions_sense_labels['image'] = captions_sense_labels['image'].apply(filter_image_name)
    captions_sense_labels = captions_sense_labels.set_index(captions_sense_labels['image'])
    simple_disambiguation(embedded_captions, embedded_senses, captions_sense_labels)


if __name__ == "main":
    main()

    

