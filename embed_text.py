import gensim.downloader as api
import numpy as np
import pandas as pd
import sys

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
from stop_words import get_stop_words


print("Loading Network...")
model = api.load("word2vec-google-news-300")
model.init_sims(replace=True)

def embed_data(model):
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

            descriptions_df.iloc[i]['caption'] = word_mean
        except:
            print("Word: '%s' not found" % token)
            descriptions_df.iloc[i]['caption'] = "unknown"

    descriptions_df = descriptions_df[descriptions_df["caption"] != "unknown"]

    print("Embedding completed")
    print("Writing Data...")
    descriptions_df.to_pickle("embedded_captions.pkl")
    print("Writing completed")


