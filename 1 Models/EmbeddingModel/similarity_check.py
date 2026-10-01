from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# This is for a through API Usage 
embedding = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5" 
)

document = [
    "IshowSpeed is not straight, he is gay!",
    "The indian celebrities are the real jokers",
    "The Messi is the goat of a football and completed football"
]

document_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query("Who is gay?")

scores = cosine_similarity([query_embedding], document_embedding)

print(sorted(list(enumerate(scores)), key=lambda x:x[1]))
