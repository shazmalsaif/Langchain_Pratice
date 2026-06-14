from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# Initialize the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Define the prompt template 
template = PromptTemplate (
    template="Write a report on this topic:{topic}",
    input_variables=["topic"]

)


# Define the prompt template 1
template1 = PromptTemplate (
    template="give me 3 line summery on this :{text}",
    input_variables=["text"]

)

prompt = template.invoke({"topic": "Artificial Intelligence"})
response = model.invoke(prompt)

prompt1 = template1.invoke({"text": response.content})
response1 = model.invoke(prompt1)

print(response1.content)