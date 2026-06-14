from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
parser = StrOutputParser()

chain = template | model | parser | template1 |model | parser


result = chain.invoke({"topic": "Artifical Intelligence"})

print(result)

