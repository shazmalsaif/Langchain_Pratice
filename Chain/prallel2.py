from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Setup Models
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)   
model1 = ChatHuggingFace(llm=llm1)

# Setup Prompts
prompt = PromptTemplate(
    template="Write information about {topic}",
    input_variables=["topic"]
)

prompt1 = PromptTemplate(
    template="Give 5 line summary of the text:{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Merge the following two texts into one: Text1:{text1} Text2:{text2}. When you are merging, mention which part is from text1 and text2.",
    input_variables=["text1", "text2"]
)
parser = StrOutputParser()

# THE SEQUENTIAL CHAIN DESIGN
chain = (
    # Step 1: Run the first prompt and store its output as 'text1'
    RunnablePassthrough.assign(
        text1=prompt | model | parser
    )
    # Step 2: Feed 'text1' into prompt1 as the 'text' variable, and store its output as 'text2'
    | RunnablePassthrough.assign(
        text2=lambda x: x["text1"],  # This maps text1 to the 'text' key prompt1 expects
    ) | RunnablePassthrough.assign(
        text2=lambda x: {"text": x["text2"]} | prompt1 | model1 | parser
    )
    # Step 3: Send both 'text1' and 'text2' into prompt2 to merge them
    | prompt2 
    | model 
    | parser
)

# Invoke the sequential chain
print(chain.invoke({'topic': 'Lord of the Mysteries'}))
