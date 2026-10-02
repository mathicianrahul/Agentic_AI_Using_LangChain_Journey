from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-4B-Instruct-2507',
    
    max_new_tokens = 100,
    temperature = 0.7
) 
model = ChatHuggingFace(llm=llm)
chat_history=[]
while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == "exit":
        break

    result= model.invoke(chat_history)
    chat_history.append(result.content)

    print("AI: ", result.content)
    #change the code 

print(chat_history)