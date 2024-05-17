import time
import requests
from qdrant_client import QdrantClient
from utils import get_text_embedding
from openai import OpenAI
import base64
from utils import generate_img_by_img

class Result:
    def __init__(self, image_data):
        self.image_data = image_data

    def write(self, file_path):
        # time.sleep(6)
        if self.image_data is None:
            raise Exception("No image data to write")
        # 解码Base64图像
        decoded_image = base64.b64decode(self.image_data)
        with open(file_path, 'wb') as f:
            f.write(decoded_image)


class ModelServer:
    def __init__(self,url):
        self.url = url

    def identify_query_and_prompt(self, message):
        client = OpenAI()
        messages = [
            {'role':'system','content':'You are a helpful assistant.'},
            {'role':'user',"content": f'{message},这个任务中检索图像过程使用的关键词是什么?只返回关键词'}
        ]
        completion = client.chat.completions.create(
            model='gpt-4o',
            messages = messages
        )
        query = completion.choices[0].message.content
        messages = [
            {'role':'system','content':'You are a helpful assistant.'},
            {'role':'user',"content": f'{message},这个任务中需要根据原图生成什么图像?只用英文返回任务要求,并强调除了图像url外不返回其它内容'}
        ]
        completion = client.chat.completions.create(
            model='gpt-4o',
            messages = messages
        )
        prompt = completion.choices[0].message.content
        return query,prompt

class RetrieverServer:
    def __init__(self, url):
        self.url = url

    def search_the_best_img(self,query):
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "CoCoImage" 
        print(f"Searching for: {query}")
        
        query_embedding = get_text_embedding(query)
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=1
        )
        
        res = []
        for hit in results:  
            res.append(hit.payload)
        image_name = res[0]['file_name']
        image_path = './coco_dataset/val2017/'+image_name
        print(image_path)
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        return image_base64

def build_rag(model, retriever):
    class RAG:
        def __init__(self, model, retriever):
            self.model = model
            self.retriever = retriever

        def query(self, query, image_only, file_type):
            # 0.确认命令
            query,prompt = self.model.identify_query_and_prompt(query)
            print(query,'*',prompt)

            # 1.检索
            image_base64 = self.retriever.search_the_best_img(query)
            # 2.生成
            #generate_info = self.model.generate_img(prompt,image_base64)
            image_data = generate_img_by_img(prompt,image_base64,)
            ret = Result(image_data)
            return ret

    return RAG(model, retriever)


