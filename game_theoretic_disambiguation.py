import numpy as np
import pandas as pd
from tqdm import tqdm


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
    train_prefix = "COCO_train2014_"
    val_prefix = "COCO_val2014_"
    if img_name.startswith(train_prefix):
        return int(img_name[len(train_prefix):-4])
    if img_name.startswith(val_prefix):
        return int(img_name[len(val_prefix):-4])
    raise ValueError('image prefix nor train and val')


def remove_duplicates(images, sense_labels):
    """
    Drop images with multiple verbs associated

    Args:
        images: A dataframe of image representations
        senses: A dataframe of senses representations
        sense_labels: A dataframe that contains the verb and the correct sense for each image

    Returns:
        A tuple filtered tuple (images, senses)
    """
    duplicates = pd.DataFrame(sense_labels.groupby("image")["lemma"].count()).query("lemma > 1")
    dups = []
    for i in range(len(duplicates)):
        dups.append(duplicates.iloc[i].name)
    images = images.drop(dups)

    for i in range(len(dups)):
        dup = dups[i]
        sense_labels = sense_labels.drop(list(sense_labels.query("image == @dup").index))

    return (images, sense_labels)


def affinity_matrix(elements):
    """
    Compute parwise similarities of a given array; such matrix is the graph weight matrix W.
    Affinity is computed through cosine similarity.

    Args:
        elements: A 1D-vector of elements of size n

    Returns:
        An nxn symmetric matrix
    """
    n = len(elements)
    W = np.zeros((n, n)) # Affinity matrix

    for i in range(n):
        w_i = elements.iloc[i]
        for j in range(i + 1, n): # Since the matrix is symmetric computations can be sped up.
            w_j = elements.iloc[j]
            w_ij = np.dot(w_i, w_j) # Cosine similarity (since vectors are already normalised).
            W[i, j] = w_ij
            W[j, i] = w_ij
    return W


def strategy_space(images, senses, sense_labels):
    """
    Generate the strategy space of the game, where rows are mixed_strategies (image representations)
    and columns are pure strategies (verb senses). Each cell is uniformly initialiased, the cells
    for which verb and sense intersection is null are set to zero.

    Args:
        images: A dataframe of image representations of size n
        senses: A dataframe of senses representations of size c
        sense_labels: A dataframe that contains the verb and the correct sense for each image

    Returns:
        An nxc matrix containing row-wise probability distributions
    """
    n = len(images)
    c = len(senses)
    S = np.zeros((n, c)) # Strategy space

    for i in range(len(images)): # rows: index of image images_captions table
        image_id = images.iloc[i].name
        verb = sense_labels.query("image == @image_id")["lemma"].iloc[0]
        filtered_senses = senses.query("lemma == @verb")
        m = len(filtered_senses) # number of senses for that verb

        for j in range(len(filtered_senses)):
            col = filtered_senses.iloc[j].name # columns: index of sense in pandas dataframe
            S[i, col] = 1 / m
    return S


def strategy_payoff(image_details, W, S, Z, images, senses, sense_labels):
    """
    Updates strategy space using Replicator Dynamics.

    Args:
        image_details: dictionary structure containing: image, senses indexes, and image absolute location
        W: Affinity matrix of size nxn
        S: Strategy space of size nxc
        Z: Payoff matrix of size cxc
        images: A dataframe of image representations of size n
        senses: A dataframe of senses representations of size c
        sense_labels: A dataframe that contains the verb and the correct sense for each image

    Returns:
        An nxc matrix containing row-wise probability distributions
    """
    pos_i = image_details.get("pos")
    verb_i_sense_idxs = image_details.get("sense_idxs")

    x_i = np.vstack(S[pos_i, verb_i_sense_idxs]) # m_i x 1 vector
    Ax = np.zeros((len(verb_i_sense_idxs), 1))

    denominator = 0
    for j in range(len(images)):
        pos_j = j
        if pos_i != pos_j:
            verb_j = sense_labels.at[images.iloc[j].name, 'lemma']
            verb_j_sense_idxs = list(senses.query("lemma == @verb_j").index)

            w_ij = W[pos_i, pos_j] # scalar
            Z_ij = Z[verb_i_sense_idxs][:, verb_j_sense_idxs] # m_i x m_j matrix
            x_j = np.vstack(S[pos_j, verb_j_sense_idxs]) # m_j x 1 vector

            Ax += w_ij * Z_ij @ x_j # Computation of fraction numerator (sum wZx) i.e. m_j x 1 vector
            denominator += x_i.T @(w_ij * Z_ij @ x_j) # scalar

    for h in range(len(x_i)):
        x_i[h] = (x_i[h] * Ax[h] / denominator)[0, 0]
    return x_i


def main():
    sense_labels = pd.read_csv("full_sense_annotations.csv")
    sense_labels = sense_labels[sense_labels['COCO/TUHOI'] == 'COCO']
    sense_labels['image'] = sense_labels['image'].apply(filter_image_name)

    images = pd.read_pickle("embedded_captions.pkl") # Images representations
    senses = pd.read_pickle("embedded_senses.pkl") # Senses representations
    
    images, sense_labels = remove_duplicates(images, sense_labels)

    sense_labels.set_index('image', inplace=True) # For performance speed-up, image_id is set as index

    # AFFINITY MATRIX
    # W = affinity_matrix(images["caption"])
    W = np.load("affinity.npy")

    # STRATEGY SPACE
    # S = strategy_space(images, senses, sense_labels)
    S = np.load("strategy_space.npy")

    # PAYOFF MATRIX
    # Z = affinity_matrix(senses["definition"])
    Z = np.load("payoff.npy")

    accuracy = [0, 0]
    for i in range(len(images)):
        image_i = images.iloc[i]

        verb_i = sense_labels.at[image_i.name, 'lemma']
        verb_i_sense_idxs = list(senses.query("lemma == @verb_i").index)
        pos_i = images.index.get_loc(image_i.name)

        # In order to reduce the number of calls to 'query' method, gathered data is passed to function as dictionary
        image_details = {"image": image_i, "sense_idxs": verb_i_sense_idxs, "pos": pos_i}

        x_t = np.vstack(S[pos_i, verb_i_sense_idxs])
        x_t1 = strategy_payoff(image_details, W, S, Z, images, senses, sense_labels)

        while np.all(np.abs(x_t1 - x_t) > 0.00001):
            x_t = x_t1
            x_t1 = strategy_payoff(image_details, W, S, Z, images, senses, sense_labels)
            S[i, verb_i_sense_idxs] = x_t1.flatten()

        pred_sense_id = senses.iloc[np.argmax(S[i])]['sense_num']
        sense_id = sense_labels.at[image_i.name, 'sense_chosen']

        if sense_id == pred_sense_id:
            accuracy[1] += 1
        else:
            accuracy[0] += 1

    accuracy = (accuracy[1] / (accuracy[0] + accuracy[1])) * 100

    print("Sense accuracy is: %s" % accuracy)


if __name__ == '__main__':
    main()
