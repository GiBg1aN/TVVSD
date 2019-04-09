import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
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
    train_prefix = 'COCO_train2014_'
    val_prefix = 'COCO_val2014_'
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
        sense_labels: A dataframe that contains the verb and the correct
            sense for each image

    Returns:
        A tuple filtered tuple (images, senses)
    """
    duplicates = pd.DataFrame(sense_labels.groupby('image')['lemma'].count()).query('lemma > 1')
    dups = []
    for i in range(len(duplicates)):
        dups.append(duplicates.iloc[i].name)
    images = images.drop(dups)

    for dup in dups:
        sense_labels = sense_labels.drop(list(sense_labels.query('image == @dup').index))

    return (images, sense_labels)


def affinity_matrix(elements):
    """
    Compute parwise similarities of a given array; such matrix is the
    graph weight matrix W.

    Affinity is computed through cosine similarity.

    Args:
        elements: A 1D-vector of elements of size n

    Returns:
        An nxn symmetric matrix
    """
    n = len(elements)
    W = np.zeros((n, n))  # Affinity matrix

    for i in range(n):
        w_i = elements.iloc[i]
        for j in range(i + 1, n):  # Since the matrix is symmetric computations can be sped up.
            w_j = elements.iloc[j]
            w_ij = np.dot(w_i, w_j)  # Cosine similarity (since vectors are already normalised).
            W[i, j] = w_ij
            W[j, i] = w_ij
    return W


def affinity_augment(affinity_matrix, mult=1):
    """ 
    Apply MinMax Scaling and Gaussian augment to affinity matrix.

    Args:
        affinity_matrix: pairwise similarity matrix
        mult: multiplier of gamma term

    Returns:
        The pairwise transformed similarity matrix
    """
    scaler = MinMaxScaler()
    scaler.fit(affinity_matrix)
    scaled_affinity = scaler.transform(affinity_matrix)
    gamma = 1 / (mult * np.var(scaled_affinity))
    augmented_affinity = np.zeros(scaled_affinity.shape)
    for i in range(len(scaled_affinity)):
        for j in range(i + 1, len(scaled_affinity)):
            a_ij = np.exp(-np.square(scaled_affinity[i, j]) * gamma) 
            augmented_affinity[i, j] = a_ij
            augmented_affinity[j, i] = a_ij
    return augmented_affinity


def strategy_space(images, senses, sense_labels):
    """
    Generate the strategy space of the game, where rows are
    mixed_strategies (image representations) and columns are pure
    strategies (verb senses). Each cell is uniformly initialiased, the
    cells for which verb and sense intersection is null are set to zero.

    Args:
        images: A dataframe of image representations of size n
        senses: A dataframe of senses representations of size c
        sense_labels: A dataframe that contains the verb and the correct
            sense for each image

    Returns:
        An nxc matrix containing row-wise probability distributions
    """
    n = len(images)
    c = len(senses)
    S = np.zeros((n, c))  # Strategy space

    for i in range(len(images)):  # Rows: index of image images_captions table
        image_id = images.iloc[i].name
        verb = sense_labels.query('image == @image_id')['lemma'].iloc[0]
        filtered_senses = senses.query('lemma == @verb')

        for j in range(len(filtered_senses)):
            col = filtered_senses.iloc[j].name  # Columns: index of sense in pandas dataframe
            S[i, col] = 1 / len(filtered_senses)
    return S


def sense_indexes_lookup(image_id, senses, sense_labels):
    """
    Lookup sense indexes in 'senses' for 'image_id'

    Args:
        image_i: input image unique index identifier
        senses: A dataframe of senses representations
        sense_labels: A dataframe that contains the verb and the correct
            sense for each image (indexed using 'image_id')

    Returns:
        A plain list of integer indexes for the strategy space
    """
    verb = sense_labels.at[image_id, 'lemma']
    verb_sense_idxs = list(senses.query('lemma == @verb').index)
    return verb_sense_idxs


def strategy_payoff(image_details, W, S, Z, images):
    """
    Play a game with each player, updating the mixed strategy of the
    input player using Replicator Dynamics.

    Args:
        image_details: dictionary structure containing:
            senses indexes, and image absolute location
        W: Affinity matrix of size nxn
        S: Strategy space of size nxc
        Z: Payoff matrix of size cxc
        images: A dataframe of image representations of size n

    Returns:
        An mx1 vector (where m is the number of senses of such verb)
        containing probability distributions of input image senses
    """
    pos_i = image_details.get('pos')
    verb_i_sense_idxs = image_details.get('sense_idxs')

    x_i = np.vstack(S[pos_i, verb_i_sense_idxs])  # m_i x 1 vector
    Ax_sum = np.zeros((len(verb_i_sense_idxs), 1))

    denominator = 0
    for pos_j, row in enumerate(images.itertuples()):
        if pos_i != pos_j:
            verb_j_sense_idxs = images.at[row.Index, 'sense_indexes']

            w_ij = W[pos_i, pos_j]  # Scalar
            Z_ij = Z[verb_i_sense_idxs][:, verb_j_sense_idxs]  # m_i x m_j matrix
            x_j = np.vstack(S[pos_j, verb_j_sense_idxs])  # m_j x 1 vector

            Ax = w_ij*Z_ij @ x_j
            Ax_sum += Ax  # Fraction numerator (sum wZx) i.e. m_j x 1 vector
            denominator += x_i.T @ Ax  # Scalar

    x_i = x_i * Ax_sum / denominator
    return x_i


def main():
    sense_labels = pd.read_csv('full_sense_annotations.csv')
    sense_labels = sense_labels[sense_labels['COCO/TUHOI'] == 'COCO']
    sense_labels['image'] = sense_labels['image'].apply(filter_image_name)

    images = pd.read_pickle('embedded_captions.pkl')  # Images representations
    senses = pd.read_pickle('embedded_senses.pkl')  # Senses representations

    images, sense_labels = remove_duplicates(images, sense_labels)

    # For performance speed up, labels the index is rebuilt once in order to use index-based access
    # instead of query access that is computational expensive. In this case image is set as index.
    #
    # For the same reason, sense_indexes for each image are computed once before looping in order
    # to avoid 'query' usage at each iteration.
    sense_labels.set_index('image', inplace=True)
    images['sense_indexes'] = images.apply(
        lambda x: sense_indexes_lookup(x.name, senses, sense_labels), axis=1)

    # W = affinity_matrix(images['caption'])  # Affinity matrix
    W = np.load('affinity.npy')  # Affinity matrix

    # S = strategy_space(images, senses, sense_labels)  # Strategy space
    S = np.load('strategy_space.npy')  # Strategy space

    # Z = affinity_matrix(senses['definition'])  # Payoff matrix
    Z = np.load('payoff.npy')  # Payoff matrix

    accuracy = [0, 0]
    for i, row in tqdm(enumerate(images.itertuples())):
        verb_i_sense_idxs = images.at[row.Index, 'sense_indexes']
        # In order to reduce the number of calls to 'query' method, gathered data is passed to
        # function as dictionary.
        image_details = {'sense_idxs': verb_i_sense_idxs, 'pos': i}

        x_t = np.vstack(S[i, verb_i_sense_idxs])
        x_t1 = strategy_payoff(image_details, W, S, Z, images)
        S[i, verb_i_sense_idxs] = x_t1.flatten()

        count = 0
        while np.all(np.abs(x_t1 - x_t) > 0.00001) and count < 1:
            x_t = x_t1
            x_t1 = strategy_payoff(image_details, W, S, Z, images)
            S[i, verb_i_sense_idxs] = x_t1.flatten()
            count += 1

        pred_sense_id = senses.iloc[np.argmax(S[i])]['sense_num']
        sense_id = sense_labels.at[row.Index, 'sense_chosen']

        if sense_id == pred_sense_id:
            accuracy[1] += 1
        else:
            accuracy[0] += 1

    accuracy = (accuracy[1] / (accuracy[0] + accuracy[1])) * 100

    print('Sense accuracy is: %s' % accuracy)
    np.save('updated_game_space.npy', S)


if __name__ == '__main__':
    main()
