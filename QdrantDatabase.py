import uuid
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models
from utils import get_text_embedding,get_image_embedding

def init_database():
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "CoCoImage"

    vector_size = 512
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )

    ids = []
    vectors = []
    payloads = []

    image_directory = "coco_dataset/val2017"
    image_list = os.listdir(image_directory)[:100]
    for idx, fname in enumerate(image_list):
        image_path = os.path.join(image_directory, fname)
        embedding = get_image_embedding(image_path)
        unique_id = str(uuid.uuid4())
        
        payload = {
            "file_name": fname,
        }
        
        ids.append(unique_id)
        vectors.append(embedding)
        payloads.append(payload)
        print(idx,len(image_list))

    batch = models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payloads
    )
    client.upsert(collection_name=collection_name, points=batch)
    print("Database initialized and data uploaded.")

def insert_img_to_collection(image_name):
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "CoCoImage"
    payload = {
            "file_name": image_name,
        }
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=get_image_embedding("coco_dataset/val2017/"+image_name),
        payload=payload
    )
    client.upsert(collection_name=collection_name, points=[point])
    print(f"Insert Image {image_name} to collection successfully!")

def search_the_best_img(query):
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "CoCoImage" 
    
    query_embedding = get_text_embedding(query)
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=1
    )
    
    res = []
    for hit in results:  
        res.append(hit.payload)
    return res

if __name__ == '__main__':
    #init_database()
    insert_img_to_collection('00000000000.jpg')
    #res = search_the_best_img('男孩')
    #print(res)
