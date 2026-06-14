from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()
# Initialize the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
# Define the prompt template

template = PromptTemplate (
    template = "Write a random person's name age and hight.\n {format_instructions}",
    input_variables = [],
    partial_variables ={"format_instructions":parser.get_format_instructions()}
)
#prompt = template.format()

chain = template | model | parser

result = chain.invoke({})

print(result)