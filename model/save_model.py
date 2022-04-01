import pickle
import pymongo
from envyaml import EnvYAML

env = EnvYAML('../resources/application.yaml')
USERNAME = env["mongodb.username"]
PASSWORD = env["mongodb.password"]

client = pymongo.MongoClient(f"mongodb+srv://{USERNAME}:{PASSWORD}@cluster0.mr8ov.mongodb.net/models?retryWrites=true&w=majority")
db = client.models

pickled_model = pickle.dumps(pickle.load(open("./finished_model.sav", "rb")))

json = {
    "model" : pickled_model,
    "name": "model"
}

db.model.update_one({'name' : "model"}, {"$set":json}, upsert=True)