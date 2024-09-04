import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# load data
with open("./data/data-500.json") as f:
    documents = json.load(f)

index_name = "movie-questions"
model_name = "multi-qa-distilbert-cos-v1"
model = SentenceTransformer(model_name)

# embedding data
print("embedding.....")
for doc in tqdm(documents):
    tsg_text = f"{doc['title']} {doc['summary']} {doc['genres'] }"
    doc['rating'] = int(doc['rating'])/10
    embed = model.encode(tsg_text)
    doc["title_summary_genres_vector"] = embed.tolist()

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "imdb_id": {"type": "text"},
            "title": {"type": "text"},
            "summary": {"type": "text"},
            "short_summary": {"type": "text"},
            "genres": {"type": "text"},
            "cast": {"type": "text"},
            "rating": {"type": "double"},
            "title_summary_genres_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}
# import data
es_client = Elasticsearch('http://localhost:9200')
es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

print("import data.....")
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
print("data initialization was done!")
