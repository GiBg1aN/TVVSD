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


def affinity_matrix(images):
    N = len(images)
    W = np.zeros((N, N)) # Affinity matrix
    for i in range(N):
        w_i = images.iloc[i].to_numpy()[0]
        for j in range(N):
            w_j = images.iloc[j].to_numpy()[0]
            W[i, j] = np.dot(w_i, w_j)
    return W


def strategy_space(images, senses):
    N = len(images)
    C = len(senses)
    S = np.zeros((N, C)) # Strategy space

    for i in range(len(images)):
        image_id = images.iloc[i].name
        verbs = labels.query('image == @image_id')["lemma"]

        for j in range(len(verbs)):
            verb = verbs.iloc[j]
            filtered_senses = senses.query("lemma == @verb")
            
            M_i = len(filtered_senses) # number of senses for that verb


    


# def main():
sense_labels = pd.read_csv("full_sense_annotations.csv")
sense_labels = sense_labels[sense_labels['COCO/TUHOI'] == 'COCO']
sense_labels['image'] = sense_labels['image'].apply(filter_image_name)


# AFFINITY MATRIX
print("Computing affinity matrix W...")
images = pd.read_pickle("embedded_captions.pkl") # Images representations
# W = affinity_matrix(images)
print("Writing W...")
# np.savetxt("affinity.csv", W, delimiter=',')
# np.save('affinity.npy', W)

# STRATEGY SPACE
senses = pd.read_pickle("embedded_senses.pkl") # Senses representations


# if __name__ == '__main__':
    # main()
