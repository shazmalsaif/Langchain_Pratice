from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm )

llm1 = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)   
model1 = ChatHuggingFace(llm=llm1 )

prompt= PromptTemplate(
    template="Write information about {topic}",
    input_variables=["topic"]
)

prompt1 = PromptTemplate(
    template ="Give 5 line summary of the text:{text}",
    input_variables =["text"]
)
prompt2 = PromptTemplate(
    template="Marge the following two text into one : Text1:{text1} Text2:{text2}. WHen you are marging mantion that which part is from part from tex1 and text 2",
    input_variables=["text1", "text2"]
)
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "text1": prompt | model | parser,
        "text2": RunnablePassthrough.assign(text= lambda x: x['topic'])|prompt1 | model1 | parser
    }
)

marge_chaiin = prompt2| model| parser

chain = parallel_chain | marge_chaiin


print(chain.invoke({'topic': 'Lord of the Mysteries '}))

