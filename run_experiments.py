from gtg import *


def combine_data(embeddings, images_features):
    full_dataframe = pd.concat([embeddings, images_features], axis=1, sort=True)
    full_dataframe['concat_image_caption'] = full_dataframe.apply(lambda r: np.concatenate([r.e_caption, r.e_image.ravel()]), axis=1)
    full_dataframe['concat_image_object'] = full_dataframe.apply(lambda r: np.concatenate([r.e_object, r.e_image.ravel()]), axis=1)
    full_dataframe['concat_image_text'] = full_dataframe.apply(lambda r: np.concatenate([r.e_combined, r.e_image.ravel()]), axis=1)
    return full_dataframe.applymap(lambda x: x / np.linalg.norm(x, ord=2))


def filter_senses(senses, sense_labels):
    """
    Remove senses that are not used in dataset images.

    Args:
        senses: A dataframe of verb senses
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).

    Returns:
        A dataset containing only the senses in 'sense_labels'
    """
    new_senses = pd.DataFrame(columns=['lemma', 'sense_num', 'definition',
                                       'ontonotes_sense_examples', 'visualness_label'])
    for _, row in enumerate(senses.itertuples()):
        sense = getattr(row, 'lemma')
        sense_id = getattr(row, 'sense_num')

        occurrences = sense_labels.query("lemma == @sense and sense_chosen == @sense_id")
        if occurrences.shape[0] > 0:
            new_senses = new_senses.append([row], sort=False)
    new_senses = new_senses.drop(columns=['Index'])
    new_senses.reset_index(inplace=True, drop=True)
    return new_senses


def run_experiment_semi_supervised(senses, sense_labels, full_features):
    y = sense_labels[['lemma', 'sense_chosen']].copy()
    y['one_hot'] = y.apply(lambda r: one_hot(senses, r.lemma, r.sense_chosen), axis=1)

    # Uniform distribution on senses related to the verb (Semi-supervised)
    seeds = [73, 37, 29, 30124, 30141, 54321, 1001001, 2051995, 579328629, 1337, 7331, 1221, 111, 99, 666]
    strategies = strategy_space(sense_labels, senses)
    for representation_type in full_features.columns.to_list():
        nodes = generate_nodes(full_features, sense_labels, representation_type)
        affinity = affinity_matrix(nodes)
        print(representation_type)
        for seed in seeds:
            labels_index = labelling(senses, sense_labels, seed)
            print('Seed: %s' % seed)
            motions, non_motions = gtg(y, affinity, labels_index, strategies)

            print("Motion: %s" % str(motions * 100))
            print("Non-motion: %s" % str(non_motions * 100))

            # with open('experiments.csv', 'w') as exps: # labels_per_class, verb_type, column, accuracy
                # exps.write('1, ' + 'motions, ' + representation_type + ', ' + str(motions) + '\n')
                # exps.write('1, ' + 'non_motions, ' + representation_type + ', ' + str(non_motions) + '\n')
                # exps.close()

def main():
    """ Run multiple GTG experiments. """
    # File reading and preprocessing
    images_features = pd.read_pickle('generated/images_features.pkl')
    images_features['e_image'] = images_features['e_image'].apply(lambda x: x / np.linalg.norm(x, ord=2))
    embedded_annotations = pd.read_pickle('generated/embedded_annotations.pkl')
    full_features = combine_data(embedded_annotations, images_features)
    embedded_senses = pd.read_pickle('generated/embedded_senses.pkl')
    senses = pd.read_csv('data/labels/verse_visualness_labels.tsv',
                         sep='\t', dtype={'sense_num': str})
    sense_labels = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv',
                               dtype={'sense_chosen': str})

    sense_labels['image'] = sense_labels['image'].apply(filter_image_name)
    sense_labels = sense_labels[sense_labels['sense_chosen'] != '-1']  # Drop unclassifiable elems
    senses.dropna(inplace=True)
    sense_labels.reset_index(inplace=True, drop=True)
    senses.reset_index(inplace=True, drop=True)

    senses['vects'] = embedded_senses['e_combined']  # Sense embeddings for prior initialisation


    # RUNS
    run_experiment_semi_supervised(filter_senses(senses, sense_labels), sense_labels, full_features)
    exit()


    # First Sense Heuristics
    print('First Sense')
    nodes = generate_nodes(full_features, sense_labels, 'e_caption')
    affinity = affinity_matrix(nodes)
    labels_index = []
    gtg(y, affinity, labels_index, strategies, True)


    # Probability based on dot prod with senses (unsupervised)
    print('Prior probability initialisation')
    for representation_type in full_features.columns.to_list():
        nodes = generate_nodes(full_features, sense_labels, representation_type)
        affinity = affinity_matrix(nodes)
        labels_index = []
        strategies = prior_knowledge(y, nodes, senses)
        print("Encoding: %s " % representation_type)
        gtg(y, affinity, labels_index, strategies)


    # Most Frequent Sense Heuristics
    print('Most Frequent Sense')
    senses = filter_senses(senses, sense_labels)
    nodes = generate_nodes(full_features, sense_labels, 'e_object')
    affinity = affinity_matrix(nodes)
    strategies = mfs_heuristic_strategies(sense_labels, senses)
    labels_index = []
    gtg(y, affinity, labels_index, strategies, True)


if __name__ == '__main__':
    main()
