from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
load_dotenv()
# Initialize the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name ='fact_1',description = 'fact 1 about the topic'),
    ResponseSchema(name ='fact_2',description = 'fact 2 about the topic'),
    ResponseSchema(name ='fact_3',description = 'fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Define the prompt template
template = PromptTemplate (
    template = "Write 3 facts about the :{topic}.\n {format_instructions}",
    input_variables = ["topic"],
    partial_variables ={"format_instructions":parser.get_format_instructions()}
)


chain =template |model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)

###[ModuleNotFoundError: No module named 'langchain.output_parsers' Need to update langchian module to the latest version to use StructuredOutputParser]