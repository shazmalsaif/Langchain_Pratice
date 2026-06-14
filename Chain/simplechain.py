from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm )

template = PromptTemplate(
    template="What is {topic}?",
    input_variables=["topic"]
)

parser = StrOutputParser()

chaiin = template| model| parser

#print(chaiin.invoke({'topic': 'AI'}))

chaiin.get_graph().print_ascii()