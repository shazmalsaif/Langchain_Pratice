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

prompt= PromptTemplate(
    template="Write information about {topic}",
    input_variables=["topic"]
)

prompt1 = PromptTemplate(
    template ="Give 5 line summary of the text:{text}",
    input_variables =["text"]
)

parser = StrOutputParser()

chaiin = prompt| model| parser | prompt1 | model | parser

print(chaiin.invoke({'topic': 'Lord of the Mysteries '}))

chaiin.get_graph().print_ascii()