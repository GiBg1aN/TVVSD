import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from stop_words import get_stop_words


def embed_data_descriptions(model):
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)
    
    print("Loading Dataset...")
    descriptions_df = pd.read_csv("filtered_annotations.csv")

    print("Dataset embedding...")
    for i in range(len(descriptions_df)):
        caption = descriptions_df.iloc[i]['caption']
        caption_tokens = [w for w in simple_preprocess(caption) if not w in stop_words]
        caption_tokens = [token for token in caption_tokens if token in model.vocab]
        
        acc = np.zeros(model.vector_size)
        for token in caption_tokens:
            acc += model.wv.word_vec(token, use_norm=True)
        word_mean = acc / len(caption_tokens)
        word_mean /= np.linalg.norm(word_mean)

        descriptions_df.iat[i, 1] = word_mean

    embedded_descriptions_df = descriptions_df.dropna()
    rows_per_image = pd.DataFrame(embedded_descriptions_df.groupby('image_id')['caption'].count())
    summed_descriptions_df = pd.DataFrame(embedded_descriptions_df.groupby('image_id')['caption'].apply(np.sum))
    mean_descriptions_df = summed_descriptions_df.apply(lambda x: x / rows_per_image.loc[x.name], axis=1)

    print("Embedding completed")
    print("Writing Data...")
    mean_descriptions_df.to_pickle("embedded_captions.pkl")
    print("Writing completed")


def embed_data_senses(model):
    stop_words = list(get_stop_words('en'))
    ntlk_stop_words = set(stopwords.words('english'))
    stop_words.extend(ntlk_stop_words)
    
    print("Loading Dataset...")
    senses_df = pd.read_csv("verse_visualness_labels.tsv", sep='\t')
    senses_df = senses_df.dropna().reset_index()

    print("Dataset embedding...")
    for i in range(len(senses_df)):
        definition = senses_df.iloc[i]['definition']
        examples = senses_df.iloc[i]['ontonotes_sense_examples']
        examples.replace('\n', ' ')

        tokens = [w for w in simple_preprocess(definition + ' ' + examples) if not w in stop_words]
        tokens = [token for token in tokens if token in model.vocab]
        
        acc = np.zeros(model.vector_size)
        for token in tokens:
            acc += model.wv.word_vec(token, use_norm=True)
        word_mean = acc / len(tokens)
        word_mean /= np.linalg.norm(word_mean)

        senses_df.iat[i, 2] = word_mean

    senses_df = senses_df.drop(["ontonotes_sense_examples", "visualness_label"], axis=1)
    senses_df = senses_df.dropna()

    print("Embedding completed")
    print("Writing Data...")
    senses_df.to_pickle("embedded_senses.pkl")
    print("Writing completed")


def main():
    # print("Loading Network...")
    # model = api.load("word2vec-google-news-300")
    # model.init_sims(replace=True)
    
    print("DESCRIPTIONS")
    embed_data_descriptions(model)
    print("SENSES")
    embed_data_senses(model)


if __name__ == '__main__':
    main()
