from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-4B-Instruct-2507',
    
    max_new_tokens = 100,
    temperature = 0.7
) 
model = ChatHuggingFace(llm=llm)
chat_history=[
    SystemMessage(content="You are a helpfull AI assistent")
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break

    result= model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print("AI: ", result.content)
    #change the code 

print(chat_history)