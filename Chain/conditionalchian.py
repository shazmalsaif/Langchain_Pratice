from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint   
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)   
model = ChatHuggingFace(llm=llm )  
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field( description="The sentiment of the feedback, either 'positive' or 'negative'.")

parser1 =PydanticOutputParser(pydantic_object=Feedback) 

prompt = PromptTemplate(
    template = "Just tell if the  feedback is possitive or negative: {feedback}. \n {format_instructions}",
    input_variables = ["feedback"],
    partial_variables={"format_instructions": parser1.get_format_instructions()}
)

 

chain = prompt | model |parser1


prompt1 =PromptTemplate(
         template="Give a 2 line  response of this possitive feedback: {feedback}",
         input_variables=["feedback"]
 )
prompt2 =PromptTemplate(
         template="Give a 2 line  response of this negative feedback: {feedback}",
         input_variables=["feedback"]
 )
branch_chain = RunnableBranch(
     (lambda x:x.sentiment == "positive", prompt1|model|parser),
     (lambda x:x.sentiment =="negative", prompt2|model|parser),
     RunnableLambda(lambda x:"No response")
  )

final_chain = chain | branch_chain

result =final_chain.invoke({"feedback":"The product is good but the delivery was late."})
print(result)

