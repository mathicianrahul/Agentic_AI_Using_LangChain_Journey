from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai',
    
    max_new_tokens = 100,
    temperature = 0.7
) 
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break

    result= model.invoke(user_input)

    print("AI: ", result.content)
    #change the code 