import pickle
import json
import numpy as np
import re
import demoji
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.en import English
import pymongo
from envyaml import EnvYAML


with open("../model/vocab2index.json") as f:
    vocab2index = json.load(f)

def encode_sentence(text, vocab2index, N=75):
    # encoded = [vocab2index.get(i, vocab2index["UNK"]) for i in cut]
    tokenized = stem_tweet(text)
    encoded = [vocab2index.get(word, vocab2index["UNK"]) for word in tokenized]
    encoded = np.zeros(N, dtype=int)
    enc = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc))
    encoded[:length] = enc[:length]
    return encoded

def stem_tweet(tweet):
    stemmer = SnowballStemmer(language='english')
    tokenized_tweets = []
    doc = tweet.split() # Tokenize tweet
    for word in doc:
        word = stemmer.stem(word) # Stem word
        tokenized_tweets.append(word)
    return tokenized_tweets

def clean_tweet(tweet):
    tok = English()
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

def evaluate_tweet(tweet):
    env = EnvYAML('../resources/application.yaml')
    USERNAME = env["mongodb.username"]
    PASSWORD = env["mongodb.password"]
    client = pymongo.MongoClient(f"mongodb+srv://{USERNAME}:{PASSWORD}@cluster0.mr8ov.mongodb.net/models?retryWrites=true&w=majority")
    db = client.models
    data = db.model.find({'name':'model'})
    model = pickle.loads(data[0]['model'])
    data = np.array(encode_sentence(clean_tweet(tweet), vocab2index=vocab2index)).reshape((1,-1))
    return bool(model.predict(data)[0])