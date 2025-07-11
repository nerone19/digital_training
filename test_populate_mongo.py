import json
from pymongo import MongoClient

MONGO_URI='mongodb://localhost:27017/'
mongo_collections = { "youtube_videos": {"name": "videos", "description": "collection containing youtube videos"}}

class MongoDB:

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MongoDB, cls).__new__(cls)
            cls._client = MongoClient(MONGO_URI)
        return cls.instance
     
    @property
    def client(self):
        return self._client


    def populate_db_from_json(self, json_file, collection_name):
        db = self._client['db']
        col = db[collection_name]
        with open(json_file, "r") as file:
            json_data = json.load(file) 
        
        for k,v in json_data.items():
            col.insert_one({k:v})

db = MongoDB()
db2 = MongoDB()
assert (db is db2)
# db.populate_db_from_json('test_res.json', mongo_collections.get("youtube_videos").get('name'))