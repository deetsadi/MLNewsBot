import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks, Response
from pydantic import BaseModel
from predict import evaluate_tweet
import json
app = FastAPI()

class Payload(BaseModel):
    text: str

@app.post("/evaluate_tweet")
def start_bot(tweet: Payload):
    try:
        if not evaluate_tweet(tweet.text):
            return json.dumps({"result" : True})
        return json.dumps({"result" : False})
    except Exception:
        return json.dumps({"result" : False})

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)