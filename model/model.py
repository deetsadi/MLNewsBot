import pandas as pd
import numpy as np
import glob
import re
import demoji
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import demoji
from sklearn.utils import shuffle, resample
import math
import spacy
from collections import Counter
import torch
from spacy.lang.en import English
import nltk
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import pickle

diction = spacy.lang.en.stop_words.STOP_WORDS
df = pd.concat([pd.read_csv(f, index_col=False) for f in glob.glob('../data/tweet_data_*.csv')],ignore_index=True, axis=0)
df = shuffle(df)

spam = df.loc[df['spam'] == 'yes']
ham = df.loc[df['spam'] =='no']

# Downsampling
spam_downsampled = resample(spam, replace=False, n_samples=10538, random_state=123) #
ham_downsampled = resample(ham, replace=False, n_samples=10538, random_state=123)
datasets = pd.concat([ham_downsampled, spam_downsampled], ignore_index=True)

# Calculate the mean of the length of tweets
datasets['number_of_words'] = datasets['tweet'].apply(lambda x: len(x.split()))
datasets['tweet_length'] = datasets['tweet'].apply(lambda x: len(x))


tok = English()
def clean_tweet(tweet):
    # Remove usernames, "RT" and Hash
    tweet = re.sub(r'(RT|[@*])(\w*)'," ", tweet)
    # Hashtags are very useful. It gives context to the tweet.
    # Remove links in tweets
    tweet = re.sub(r'http\S+', " ", tweet)
    # We remove "#" and keep the tags
    tweet = re.sub(r'(\\n)|(\#)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])','', tweet)
    tweet = re.sub(r'(<br\s*/><br\s*/>)|(\-)|(\/)', " ", tweet)
    # convert to lower case
    tweet = re.sub(r"[^a-zA-Z0-9]", " ", tweet.lower()) # Convert to lower case
    # Tweets are usually full of emojis. We need to remove them.
    tweet = demoji.replace(tweet, repl="")
    # Stop words don't meaning to tweets. They can be removed
    tweet_words = tok(tweet)
    clean_tweets = []
    for word in tweet_words:
        if word.is_stop==False and len(word) > 1:
            clean_tweets.append(word.text.strip())

    tweet = " ".join(clean_tweets)
    
    return tweet

datasets['tweet'] = datasets['tweet'].apply(lambda x: clean_tweet(x)) # Clean tweet
shuffle(datasets) # Shuffle datasets


# Encode categorical variables
encoder = LabelEncoder()
spam_encoded = encoder.fit_transform(datasets['spam'])

spam_index_mapping = {index: label for index, label in 
                  enumerate(encoder.classes_)}
datasets['label'] = spam_encoded

datasets['number_of_words'] = datasets['tweet'].apply(lambda x: len(x.split()))
datasets['tweet_length'] = datasets['tweet'].apply(lambda x: len(x))


stemmer = SnowballStemmer(language='english')
def stem_tweet(tweet):
    tokenized_tweets = []
    doc = tweet.split()
    for word in doc:
        word = stemmer.stem(word) 
        tokenized_tweets.append(word)
    return tokenized_tweets

def save_dict(filename, data):
    json.dump(data, open( filename, 'w' ) )

counts = Counter()
for index, row in datasets.iterrows():
  counts.update(stem_tweet(row['tweet']))

for word in list(counts):
    if(counts[word] < 2):
        del counts[word]

vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]

for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

save_dict("words.json", words)
save_dict("vocab2index.json", vocab2index)
def encode_sentence(text, vocab2index, N=75):
    # encoded = [vocab2index.get(i, vocab2index["UNK"]) for i in cut]
    tokenized = stem_tweet(text)
    encoded = [vocab2index.get(word, vocab2index["UNK"]) for word in tokenized]


    encoded = np.zeros(N, dtype=int)
    enc = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc))
    encoded[:length] = enc[:length]
    return encoded

datasets['encoded_tweet'] = datasets['tweet'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))

tweet_datasets = datasets.loc[datasets['number_of_words'] > 0]
tweet_datasets = tweet_datasets.reset_index()

X = list(tweet_datasets['encoded_tweet'])
y = list(tweet_datasets['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert (len(X_test)+ len(X_train) ==  tweet_datasets['encoded_tweet'].count())

from xgboost import XGBClassifier

clf = XGBClassifier()
clf.fit(X_train, y_train)

pickle.dump(clf, open("finished_model.sav", "wb"))