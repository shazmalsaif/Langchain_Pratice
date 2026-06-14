from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the Hugging Face endpoint and model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 2. Define the schema FIRST
schema = [
    ResponseSchema(name='fact_1', description='fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='fact 3 about the topic')
]

# 3. Initialize the parser AFTER the schema is defined
parser = StructuredOutputParser.from_response_schemas(schema)

# 4. Define the prompt template (Note: removed the accidental colon before {topic})
template = PromptTemplate(
    template="Write 3 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 5. Build the LCEL chain
# LangChain handles the flow: Inputs -> Template -> Model -> Parser
chain = template | model | parser

# 6. Invoke the chain by passing the required variables
result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)