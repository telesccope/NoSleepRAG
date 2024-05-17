from model import *

retriever = RetrieverServer(url='localhost:6333')
model = ModelServer(url='localhost:9081/v1/chat')
rag = build_rag(model,retriever)

if __name__ == '__main__':
    while True:
        q = input('>>')
        ret = rag.query(q,image_only=True, file_type='JPEG')
        ret.write('./output.jpeg')



