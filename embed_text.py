import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import gensim.downloader as api
from gensim.utils import simple_preprocess
from stop_words import get_stop_words
from typos import TYPOS, TYPOS2


def preprocess_text(caption, model, stop_words):
    """
    Tokenize and remove stopwords/unknown words from a string 'caption'

    Args:
        caption: input string
        model: word2vec model
        stop_words: list of stopwords

    Returns:
        A list of filtered words
    """
    caption_tokens = [w for w in simple_preprocess(caption) if not w in stop_words]
    return [token for token in caption_tokens if token in model.vocab]


def embed_text(text_tokens, model):
    """
    Embed a sequence of words and average and normalise them

    Args:
        text_tokens: tokenised string
        model: word2vec model
    """
    acc = np.zeros(model.vector_size)
    for token in text_tokens:
        acc += model.wv.word_vec(token, use_norm=True)
    word_average = acc / len(text_tokens)

    return word_average / np.linalg.norm(word_average)


def embed_data_descriptions(model, input_df):
    """
    Embed image descriptions into 300-dim vectors using word2vec
    embedding.

    Args:
        model: pre-trained word2vec gensim model
        input_df: dataframe of image captions with columns:
            image_id, category, caption

    Returns:
        A dataframe of numpy 300-dim vectors with image id as index
    """

    descriptions_df = input_df.copy()

    # Stopwords definition
    stop_words = list(get_stop_words('en'))
    stop_words.extend(set(stopwords.words('english')))

    # Caption preprocessing
    captions_tokens = descriptions_df['caption'].apply(
        lambda r: preprocess_text(r, model, stop_words))
    categories_tokens = descriptions_df['category'].apply(
        lambda r: preprocess_text(r, model, stop_words))

    # Caption embedding
    descriptions_df['e_caption'] = captions_tokens.apply(lambda r: embed_text(r, model))
    descriptions_df['e_category'] = categories_tokens.apply(lambda r: embed_text(r, model))
    descriptions_df['e_combined'] = (captions_tokens + categories_tokens).apply(
        lambda r: embed_text(r, model))

    descriptions_df.drop(['caption', 'category'], axis=1, inplace=True)

    # Captions of the same image are averaged
    group_captions = descriptions_df.groupby('image_id')['e_caption']
    group_categories = descriptions_df.groupby('image_id')['e_category']
    group_combined = descriptions_df.groupby('image_id')['e_combined']
    rows_per_image = pd.DataFrame(group_captions.count())
    accumulator_captions = group_captions.apply(np.sum).to_frame()
    accumulator_categories = group_categories.apply(np.sum).to_frame()
    accumulator_combined = group_combined.apply(np.sum).to_frame()
    summed = pd.concat([accumulator_captions, accumulator_categories, accumulator_combined], axis=1)

    unnormalised_average = summed.divide(rows_per_image.values, axis=0)
    return unnormalised_average.applymap(lambda r: r / np.linalg.norm(r))


def embed_data_senses(model, input_df):
    """
    Embed verb senses into 300-dim vectors using word2vec embedding.

    Args:
        model: pre-trained word2vec gensim model
        input_df: dataframe of senses definitions with columns:
            lemma, sense_num, definition, ontonotes_sense_examples

    Returns:
        A dataframe with columns: lemma, sense_num, definition; where
        the definition is a numpy 300-dim vector
    """
    # Stopwords definition
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)

    senses_df = input_df.dropna().reset_index(drop=True)

    # Senses preprocessing
    definitions_tokens = senses_df['definition'].apply(
        lambda r: preprocess_text(r, model, stop_words))
    examples_tokens = senses_df['ontonotes_sense_examples'].apply(
        lambda r: preprocess_text(r.replace('\n', ' '), model, stop_words))

    # Senses embedding
    combined_tokens = definitions_tokens + examples_tokens
    senses_df['e_definition'] = definitions_tokens.apply(lambda r: embed_text(r, model))
    senses_df['e_examples'] = examples_tokens.apply(lambda r: embed_text(r, model))
    senses_df['e_combined'] = combined_tokens.apply(lambda r: embed_text(r, model))

    senses_df.drop(
        ['definition', 'ontonotes_sense_examples', 'visualness_label'], axis=1, inplace=True)

    return senses_df


def spell_fix(filepath, typos):
    """
    Correct 'filepath' text using 'typos'.

    Args:
        filepath: path of file to fix
        typos: list of tuples ('wrong', 'correct')

    Returns:
        None (Write the fixed test in a file _filepath)
    """
    with open(filepath, 'r') as file:
        data = file.read()
        for typo, correct in typos:
            data = data.replace(typo, correct)
    with open('_' + filepath, 'w') as file:
        file.write(data)


def main():
    print('Loading word2vec Network...')
    model = api.load('word2vec-google-news-300')
    model.init_sims(replace=True)

    print('Spellchecking annotations...')
    spell_fix('filtered_annotations.csv', TYPOS)
    print('Embedding annotations...')
    embedded_captions = embed_data_descriptions(model, pd.read_csv('_filtered_annotations.csv'))
    print('Writing Data...')
    embedded_captions.to_pickle('embedded_captions.pkl')

    print('Spellchecking senses...')
    spell_fix('verse_visualness_labels.tsv', TYPOS2)
    print('Embedding senses...')
    embedded_senses = embed_data_senses(
        model, pd.read_csv('_verse_visualness_labels.tsv', sep='\t'))
    print('Writing Data...')
    embedded_senses.to_pickle('embedded_senses.pkl')


if __name__ == '__main__':
    main()
