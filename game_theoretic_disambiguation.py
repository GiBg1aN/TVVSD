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
        for j in range(i + 1, N):
            w_j = images.iloc[j].to_numpy()[0]
            w_ij = np.dot(w_i, w_j)
            W[i, j] = w_ij
            W[j, i] = w_ij
    return W


def strategy_space(images, senses, sense_labels):
    n = len(images)
    q = len(senses)
    S = np.zeros((n, q)) # Strategy space

    for i in range(len(images)): # rows: index of image images_captions table
        image_id = images.iloc[i].name
        verb = sense_labels.query("image == @image_id")["lemma"].iloc[0]
        filtered_senses = senses.query("lemma == @verb")
        M_p = len(filtered_senses) # number of senses for that verb

        for j in range(len(filtered_senses)): 
            col = filtered_senses.iloc[j].name # columns: index of sense in pandas dataframe
            S[i, col] = 1 / M_p
    return S
    


# def main():
sense_labels = pd.read_csv("full_sense_annotations.csv")
sense_labels = sense_labels[sense_labels['COCO/TUHOI'] == 'COCO']
sense_labels['image'] = sense_labels['image'].apply(filter_image_name)

images = pd.read_pickle("embedded_captions.pkl") # Images representations

duplicates = pd.DataFrame(sense_labels.groupby("image")["lemma"].count()).query("lemma > 1")

dups = []
for i in range(len(duplicates)):
    dups.append(duplicates.iloc[i].name)
images = images.drop(dups)

for i in range(len(dups)):
    dup = dups[i]
    sense_labels = sense_labels.drop(list(sense_labels.query("image == @dup").index))

senses = pd.read_pickle("embedded_senses.pkl").reset_index() # Senses representations


# AFFINITY MATRIX
# W = affinity_matrix(images)
W = np.load("affinity.npy")

# STRATEGY SPACE
# S = strategy_space(images, senses, sense_labels)
S = np.load("strategy_space.npy")

# Affinity matrix
# Z = affinity_matrix(senses)
Z = np.load("payoff.npy")

def strategy_payoff(image_i, W, S, Z, images, senses, sense_labels):
    idx_i = image_i.name
    pos_i = images.index.get_loc(idx_i)
    verb_i = sense_labels.query("image == @idx_i")["lemma"].iloc[0]
    verb_i_sense_idxs = list(senses.query("lemma == @verb_i").index)
    x_i = np.vstack(np.trim_zeros(S[pos_i, :]))

    Ax = np.zeros((len(verb_i_sense_idxs), 1))
    den = 0

    for j in range(len(images)):
        image_j = images.iloc[j]
        idx_j = image_j.name
        pos_j = j
        verb_j = sense_labels.query("image == @idx_j")["lemma"].iloc[0]
        verb_j_sense_idxs = list(senses.query("lemma == @verb_j").index)

        w_ij = W[pos_i, pos_j]
        Z_ij = Z[verb_i_sense_idxs][:, verb_j_sense_idxs]
        x_j = np.vstack(np.trim_zeros(S[pos_j, :]))

        Ax += w_ij * Z_ij @ x_j
        den += x_i.T @(w_ij * Z_ij @ x_j)

    for h in range(len(x_i)):
        x_i[h] = ((x_i[h] * Ax[h]) / den)[0, 0]
    return x_i
        



# image_i = images.iloc[73]

# for i in range(len(images)):
    # image_i = images.iloc[i]
    # strategy_payoff(image_i, W, S, Z, images, senses, sense_labels)



# if __name__ == '__main__':
    # main()
