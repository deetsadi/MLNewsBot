import tweepy
import uvicorn
import time
import sys
from envyaml import EnvYAML
from fastapi import FastAPI, Request, BackgroundTasks, Response
from pydantic import BaseModel
from predict import evaluate_tweet
import preprocessor as p
import requests
import json


env = EnvYAML('../resources/application.yaml')
CONSUMER_KEY = env["twitter.consumer_key"]
CONSUMER_KEY_SECRET = env["twitter.consumer_key_secret"]
ACCESS_TOKEN = env["twitter.access_token"]
ACCESS_TOKEN_SECRET = env["twitter.access_token_secret"]

app = FastAPI()

class Payload(BaseModel):
    data: str

class StdOutListener(tweepy.Stream):
    ''' Handles data received from the stream. '''
    
    time_between_tweets = 3600
    previous_text = ""

    saved_time = time.time()
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_KEY_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    p.set_options(p.OPT.MENTION)
    
    def on_status(self, status):
        # Prints the text of the tweet
        status.text = p.clean(status.text)

        # print (status.text)
        # print (evaluate_tweet(status.text))
        # print ("______________")
        if self.valid_retweet(status.text):
            #self.api.update_status(status.text)
            self.api.retweet(status.id)
            self.saved_time = time.time()
            self.previous_text = status.text
        
        return True
    
    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return False # To continue listening
    
    def on_timeout(self):
        print('Timeout...')
        return False # To continue listening
    
    def valid_retweet(self, tweet):
        if time.time() - self.saved_time >= self.time_between_tweets and tweet != self.previous_text:
            headers = {"Content-Type": "application/json"}
            payload = {"text":tweet}
            result = requests.post("https://tweet-evaluation-api.herokuapp.com/evaluate_tweet", headers=headers, data = json.dumps(payload))
            return not json.loads(result.json())["result"]
    
    def set_time_between_tweets(self, time):
        self.time_between_tweets = time
    
    def get_time_between_tweets(self):
        return self.time_between_tweets


myStream = StdOutListener(consumer_key=CONSUMER_KEY, \
            consumer_secret=CONSUMER_KEY_SECRET, \
            access_token=ACCESS_TOKEN, \
            access_token_secret=ACCESS_TOKEN_SECRET)


@app.post("/start_bot")
async def start_bot(background_tasks: BackgroundTasks):
    global myStream
    try:
        background_tasks.add_task(myStream.filter, track=['#python'])
        return "Success! Bot has been started."
    except Exception:
        return "Failed! Bot was not started."


@app.post("/stop_bot")
def stop_bot():
    global myStream
    try:
        myStream.disconnect()
        return "Success! Bot was stopped."
    except Exception:
        return "Failed! Bot was not stopped."

@app.get("/get_time_between_tweets")
def get_time_between_tweets():
    global myStream
    return f"Time between tweets is {myStream.get_time_between_tweets()} seconds"

@app.put("/set_time_between_tweets")
def set_time_between_tweets(time: Payload):
    global myStream
    myStream.set_time_between_tweets(int(time.data))
    return f"Time between tweets has been changed to {time.data} seconds"

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)