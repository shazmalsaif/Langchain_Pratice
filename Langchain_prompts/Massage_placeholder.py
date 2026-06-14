from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation",
    max_new_tokens = 512,
    stop_sequences = ["\nuser:" "<|im_end|>", "User:"]
)   
model = ChatHuggingFace(llm = llm) 
#chat prompt template

Chat_template = ChatPromptTemplate.from_messages([
    ('system','you are a helpful assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{input}')
])
chat_history = []
#load ChatHIstory
with open("Langchain_prompts/Chatbot.txt") as f:
    chat_history.extend([HumanMessage(content=line.strip()) for line in f.readlines() if line.strip()])

#Creat propmt
prompt =Chat_template.invoke({'chat_history':chat_history, 'input':'where is my refund?'})

#result = model.invoke(prompt)
#print(result.content)
print(prompt)