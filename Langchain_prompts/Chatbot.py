from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation",
    max_new_tokens = 512,
    stop_sequences = ["\nuser:" "<|im_end|>", "User:"]
)   

model = ChatHuggingFace(llm = llm)

Chat_history = [
    SystemMessage(content = "You are a helpful assistant.")

]


while True:
    User_input = input("User: ")
    if User_input.lower() == 'exit':
        print('Exititing')
        break
    else: Chat_history.append(HumanMessage(content = User_input))

    result = model.invoke(Chat_history)

    Chat_history.append(AIMessage(content = result.content))

    print(result.content)

print(Chat_history)