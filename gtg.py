import numpy as np
import pandas as pd
from visual_verb_disambiguation import filter_image_name


def filter_senses(senses, sense_labels):
    new_senses = pd.DataFrame(columns=['lemma', 'sense_num', 'definition', 'ontonotes_sense_examples', 'visualness_label'])
    for _, row in enumerate(senses.itertuples()):
        sense = getattr(row, 'lemma')
        sense_id = getattr(row, 'sense_num')

        occurrences = sense_labels.query("lemma == @sense and sense_chosen == @sense_id")
        if len(occurrences) > 0:
            new_senses = new_senses.append([row], sort=False)
    new_senses = new_senses.drop(columns=['Index'])
    new_senses.reset_index(inplace=True, drop=True)
    return new_senses


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
        w_i = elements[i]
        for j in range(i + 1, n):  # Since the matrix is symmetric computations can be sped up.
            w_j = elements[j]
            w_ij = np.dot(w_i, w_j)  # Cosine similarity (since vectors are already normalised).
            W[i, j] = w_ij
            W[j, i] = w_ij
    return W


def mfs_heuristic_strategies(sense_labels, senses):
    n = len(sense_labels)
    c = len(senses)
    S = np.zeros((n, c))  # Strategy space

    for i, row in enumerate(sense_labels.itertuples()):  # Rows: index of image images_captions table
        verb = getattr(row, 'lemma')
        filtered_senses = senses.query('lemma == @verb') 
        mfs_idx = sense_labels.query("lemma == @verb").groupby('sense_chosen')['lemma'].count().values.argmax()

        col = filtered_senses.iloc[mfs_idx].name  # Columns: index of sense in pandas dataframe
        S[i, col] = 1 
    return S


def strategy_space(sense_labels, senses):
    # """
    # Generate the strategy space of the game, where rows are
    # mixed_strategies (image representations) and columns are pure
    # strategies (verb senses). Each cell is uniformly initialiased, the
    # cells for which verb and sense intersection is null are set to zero.

    # Args:
        # images: A dataframe of image representations of size n
        # senses: A dataframe of senses representations of size c
        # sense_labels: A dataframe that contains the verb and the correct
            # sense for each image

    # Returns:
        # An nxc matrix containing row-wise probability distributions
    # """
    n = len(sense_labels)
    c = len(senses)
    S = np.zeros((n, c))  # Strategy space

    for i, row in enumerate(sense_labels.itertuples()):  # Rows: index of image images_captions table
        verb = getattr(row, 'lemma')
        filtered_senses = senses.query('lemma == @verb')

        for j in range(len(filtered_senses)):
            col = filtered_senses.iloc[j].name  # Columns: index of sense in pandas dataframe
            S[i, col] = 1 / len(filtered_senses)
    return S


def player_strategy_indexing(row_index, verb, senses):
    column_indexes = senses.query("lemma == @verb").index.tolist()
    return row_index, column_indexes


def prior_knowledge(y, nodes, senses, strategies):
    p = strategies.copy()
    for i, row in enumerate(y.itertuples()):
        i_t = nodes[i]
        verb = getattr(row, 'lemma')
        filtered_senses = senses.query('lemma == @verb')
        dot_prod = filtered_senses['vects'].apply(lambda s_t: np.dot(i_t, s_t)).to_numpy()
        probs = dot_prod / dot_prod.sum()
        p[player_strategy_indexing(i, verb, senses)] = probs

    return p


def one_hot(senses, verb, sense_num):
    sense = senses.query("lemma == @verb and sense_num == @sense_num").iloc[0]
    vector = np.zeros(len(senses))
    vector[senses.index.get_loc(sense.name)] = 1
    return vector


def generate_nodes(images, labels, representation_type):
    nodes = None
    for i, row in enumerate(labels.itertuples()):
        img_name = getattr(row, 'image')
        if nodes is None:
            nodes = images.loc[img_name][representation_type]
        else:
            node = images.loc[img_name][representation_type]
            nodes = np.vstack([nodes, node])
    return nodes


def labelling(senses, sense_labels, seed):
    myseed = np.random.RandomState(seed)
    labels = []
    for i, row in enumerate(senses.itertuples()):
        verb = getattr(row, 'lemma')
        sense_num = getattr(row, 'sense_num')
        node = sense_labels.query("lemma == @verb and sense_chosen == @sense_num").sample(n=1, random_state=myseed).iloc[0]
        node_idx = node.name
        labels.append(node_idx)
    return np.array(labels)


# One-hot encoded senses
def gtg(y_, W, labeled_senses_idx, strategies, first_sense=False):

    y = y_.copy()
    n_points = len(W)
    n_senses = strategies.shape[1]

    p = strategies.copy()

    y['one_hot'] = y.apply(lambda r: one_hot(senses, r.lemma, r.sense_chosen), axis=1)
    plabs = y['one_hot'].iloc[labeled_senses_idx]
    for i in range(len(labeled_senses_idx)):
        p[labeled_senses_idx[i], :] = plabs.iloc[i] 

    # p[labeled_senses_idx, :] = plabs

    n_iter = 0
    if first_sense:
        p_new = p
    else:
        p_new = np.zeros((n_points, n_senses))
        while True:
            q = W @ p
            dummy = p * q

            for k in range(n_senses):
                p_new[:, k] = dummy[:, k] / np.sum(dummy, 1)

            diff = np.linalg.norm(p[:] - p_new[:])
            p = p_new
            n_iter += 1

            if diff < 10 ** -4 or n_iter == 10 ** 4:
                break;

    unlabeled = np.setdiff1d(np.arange(n_points), labeled_senses_idx)
    n_unlabeled = len(unlabeled)

    correct = {'motion': 0, 'non_motion': 0}
    wrong = {'motion': 0, 'non_motion': 0}
    y_cap = np.zeros((n_unlabeled, 1))

    # Accuracy statistics
    for i in range(n_unlabeled):
        y_cap[i] = np.argmax(p_new[unlabeled[i], :])

        if y['lemma'].iloc[unlabeled[i]] in verb_types['motion']:
            if y_cap[i] == np.argmax(y['one_hot'].iloc[unlabeled[i]]):
                correct['motion'] += 1
            else:
                wrong['motion'] += 1
        elif y['lemma'].iloc[unlabeled[i]] in verb_types['non_motion']:
            if y_cap[i] == np.argmax(y['one_hot'].iloc[unlabeled[i]]):
                correct['non_motion'] += 1
            else:
                wrong['non_motion'] += 1
        else:
            raise ValueError('Unknown verb type')


    print("Motion: %s" % (correct['motion'] / (correct['motion'] + wrong['motion'])))
    print("Non-motion: %s" % (correct['non_motion'] / (correct['non_motion'] + wrong['non_motion'])))


verb_types = {}

with open('data/labels/motion_verbs.csv') as motion_verbs:
    verb_types['motion'] = [line.rstrip('\n') for line in motion_verbs]

with open('data/labels/non_motion_verbs.csv') as non_motion_verbs:
    verb_types['non_motion'] = [line.rstrip('\n') for line in non_motion_verbs]


sense_labels = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv', dtype={'sense_chosen': str})
sense_labels['image'] = sense_labels['image'].apply(filter_image_name)
sense_labels = sense_labels[sense_labels['sense_chosen'] != '-1']
sense_labels.reset_index(inplace=True, drop=True)
senses = pd.read_csv('data/labels/verse_visualness_labels.tsv', sep='\t', dtype={'sense_num': str})
senses.dropna(inplace=True)
senses.reset_index(inplace=True, drop=True)
embedded_annotations = pd.read_pickle('generated/embedded_annotations.pkl')
embedded_senses = pd.read_pickle('generated/embedded_senses.pkl')

senses['vects'] = embedded_senses['e_combined']

y = sense_labels[['lemma','sense_chosen']]
# Uniform distribution on senses related to the verb (Semi-supervised)
seeds = [1001001, 2051995, 579328629, 1337, 7331, 1221, 111, 99, 666, 73, 37, 29, 30124, 30141, 54321]
senses = filter_senses(senses, sense_labels)
for seed in seeds:
    print('Seed: %s' % seed)
    for representation_type in embedded_annotations.columns.to_list():
        nodes = generate_nodes(embedded_annotations, sense_labels, representation_type)
        W = affinity_matrix(nodes)
        S = strategy_space(sense_labels, senses)
        labels_index = labelling(senses, sense_labels, seed)
        print("Encoding: %s " % representation_type)
        gtg(y, W, labels_index, S)


# Probability based on dot prod with senses (unsupervised)
senses = filter_senses(senses, sense_labels)
print('Prior probability initialised')
for representation_type in embedded_annotations.columns.to_list():
    nodes = generate_nodes(embedded_annotations, sense_labels, representation_type)
    W = affinity_matrix(nodes)
    S = strategy_space(sense_labels, senses)
    labels_index = []
    S = prior_knowledge(y, nodes, senses, S)
    print("Encoding: %s " % representation_type)
    gtg(y, W, labels_index, S)


# First Sense Heuristics
nodes = generate_nodes(embedded_annotations, sense_labels, 'e_caption')
W = affinity_matrix(nodes)
S = strategy_space(sense_labels, senses)
labels_index = []
gtg(y, W, labels_index, S, True)


# Most Frequent Sense Heuristics
senses = filter_senses(senses, sense_labels)
nodes = generate_nodes(embedded_annotations, sense_labels, 'e_object')
W = affinity_matrix(nodes)
S = mfs_heuristic_strategies(sense_labels, senses)
labels_index = []
gtg(y, W, labels_index, S, True)

