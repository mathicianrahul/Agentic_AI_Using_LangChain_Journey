from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()
# This is for a through API Usage 
embedding = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5" 
)


result = embedding.embed_query("what is the capital of India?")
print(result)