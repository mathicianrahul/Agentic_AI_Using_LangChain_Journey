from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-4B-Instruct-2507',
    max_new_tokens=100,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a Helpfull assistent !"),
    HumanMessage(content="What is the langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)