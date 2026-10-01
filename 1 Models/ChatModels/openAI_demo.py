from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-5.4-mini', temperature=1) #Tempture is a parameter 

result = model.invoke("What is capital of India?")
# print(result)
print(result.content)