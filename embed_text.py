import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import gensim.downloader as api
from gensim.utils import simple_preprocess
from stop_words import get_stop_words


def embed_data_descriptions(model, descriptions_df):
    """
    Embed image descriptions into 300-dim vectors using word2vec
    embedding.

    Args:
        model: pre-trained word2vec gensim model
        descriptions_df: dataframe of image captions with columns:
            caption, image_id

    Returns:
        A dataframe of numpy 300-dim vectors with image id as index
    """
    # Stopwords definition
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)

    for i in range(len(descriptions_df)):
        # Caption preprocessing
        caption = descriptions_df.iloc[i]['caption']
        caption_tokens = [w for w in simple_preprocess(caption) if not w in stop_words]
        filtered_tokens = [token for token in caption_tokens if token in model.vocab]

        # Caption embedding
        acc = np.zeros(model.vector_size)
        for token in filtered_tokens:
            acc += model.wv.word_vec(token, use_norm=True)
        word_average = acc / len(filtered_tokens)

        descriptions_df.iat[i, 1] = word_average / np.linalg.norm(word_average)

    # Captions of the same image are averaged
    grouped_captions = descriptions_df.groupby('image_id')['caption']
    rows_per_image = pd.DataFrame(grouped_captions.count())
    accumulator_df = pd.DataFrame(grouped_captions.apply(np.sum))
    average_descriptions_df = accumulator_df.apply(lambda x: x / rows_per_image.loc[x.name], axis=1)

    return average_descriptions_df


def embed_data_senses(model, senses_df):
    """
    Embed verb senses into 300-dim vectors using word2vec embedding.

    Args:
        model: pre-trained word2vec gensim model
        senses_df: dataframe of senses definitions with columns:
            lemma, sense_num, definition, ontonotes_sense_examples

    Returns:
        A dataframe with columns: lemma, sense_num, definition; where
        the definition is a numpy 300-dim vector
    """
    # Stopwords definition
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)

    senses_df = senses_df.dropna().reset_index(drop=True)

    for i in range(len(senses_df)):
        # Text preprocessing
        definition = senses_df.iloc[i]['definition']
        examples = senses_df.iloc[i]['ontonotes_sense_examples']
        examples.replace('\n', ' ')
        tokens = [w for w in simple_preprocess(definition + ' ' + examples) if not w in stop_words]
        filtered_tokens = [token for token in tokens if token in model.vocab]

        # Sense embedding
        acc = np.zeros(model.vector_size)
        for token in filtered_tokens:
            acc += model.wv.word_vec(token, use_norm=True)
        word_average = acc / len(filtered_tokens)

        senses_df.iat[i, 2] = word_average / np.linalg.norm(word_average)
    embedded_senses_df = senses_df.drop(['ontonotes_sense_examples', 'visualness_label'], axis=1)

    return embedded_senses_df


def main():
    print('Loading word2vec Network...')
    model = api.load('word2vec-google-news-300')
    model.init_sims(replace=True)

    print('DESCRIPTIONS')
    embedded_captions = embed_data_descriptions(model, pd.read_csv('filtered_annotations.csv'))
    print('Embedding completed')
    print('Writing Data...')
    embedded_captions.to_pickle('embedded_captions.pkl')
    print('Writing completed')

    print('SENSES')
    embedded_senses = embed_data_senses(model, pd.read_csv('verse_visualness_labels.tsv', sep='\t'))
    print('Embedding completed')
    print('Writing Data...')
    embedded_senses.to_pickle('embedded_senses.pkl')
    print('Writing completed')


if __name__ == '__main__':
    main()
