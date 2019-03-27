import gensim.downloader as api
import numpy as np
import pandas as pd

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
from stop_words import get_stop_words


print("Loading Network...")
model = api.load("word2vec-google-news-300")
model.init_sims(replace=True)

def embed_data_descriptions(model):
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)
    
    print("Loading Dataset...")
    descriptions_df = pd.read_csv("filtered_annotations.csv")
    descriptions_df = pd.DataFrame(descriptions_df.groupby('image_id')['caption'].apply(lambda x: "%s" % ' '.join(x)))

    print("Dataset embedding...")
    for i in range(len(descriptions_df)):
        caption = descriptions_df.iloc[i]['caption']
        caption_tokens = simple_preprocess(remove_stopwords(caption))
        caption_tokens = [w for w in caption_tokens if not w in stop_words]
        
        acc = np.zeros(model.vector_size)
        try:
            for token in caption_tokens:
                acc += model.wv.word_vec(token, use_norm=True)
            word_mean = acc / len(caption_tokens)

            descriptions_df.at[i, 'caption'] = None
            descriptions_df.at[i, 'caption'] = word_mean
        except (KeyError):
            print("Word: '%s' not found" % token)
            descriptions_df.at[i, 'caption'] = None

    descriptions_df = descriptions_df.dropna()

    print("Embedding completed")
    print("Writing Data...")
    descriptions_df.to_pickle("embedded_captions.pkl")
    print("Writing completed")


def embed_data_senses(model):
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)
    
    print("Loading Dataset...")
    senses_df = pd.read_csv("verse_visualness_labels.tsv", sep='\t')
    senses_df = senses_df.dropna()

    print("Dataset embedding...")
    for i in range(len(senses_df)):
        definition = senses_df.iloc[i]['definition']
        examples = senses_df.iloc[i]['ontonotes_sense_examples']
        examples.replace('\n', '\. ')
        examples_tokens = simple_preprocess(remove_stopwords(examples))
        examples_tokens = [w for w in examples_tokens if not w in stop_words]

        use_examples = True
        for token in examples_tokens:
            if token not in model.vocab:
                use_examples = False

        if use_examples:
            tokens = simple_preprocess(remove_stopwords(definition + ' ' + examples))
        else:
            tokens = simple_preprocess(remove_stopwords(definition))
        tokens = [w for w in tokens if not w in stop_words]
        
        acc = np.zeros(model.vector_size)
        try:
            for token in tokens:
                acc += model.wv.word_vec(token, use_norm=True)
            word_mean = acc / len(tokens)

            senses_df.at[i, 'definition'] = None
            senses_df.at[i, 'definition'] = word_mean
        except (KeyError):
            print("Word: '%s' not found" % token)
            senses_df.at[i, 'definition'] = None

    senses_df = senses_df.dropna()
    senses_df = senses_df.drop("visualness_label", axis=1)

    print("Embedding completed")
    print("Writing Data...")
    senses_df.to_pickle("embedded_senses.pkl")
    print("Writing completed")


