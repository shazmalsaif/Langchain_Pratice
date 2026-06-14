from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
# Initialize the Hugging Face endpoint  
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class prompt(BaseModel):
    fact1: str = Field(description ="Fact 1 about the topic")
    fact2: str = Field(description ="Fact 2 about the topic")
    fact3: str = Field(description = "Fact 3 about the topic")

parser = PydanticOutputParser(pydantic_object=prompt)

template = PromptTemplate (
    template = "Write 3 fact about the {topic}.\n {format_instructions}",
    input_variables =["topic"],
    partial_variables ={"format_instructions":parser.get_format_instructions()}
)

chain = template | model| parser

result = chain.invoke({"topic": "Langchain"})

#print(result.model_dump_json(indent=4))
print(result.model_dump())