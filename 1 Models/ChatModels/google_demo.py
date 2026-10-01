from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')
result = model.invoke('Prove that the square root of 2 is irrational.')

print(result.text)