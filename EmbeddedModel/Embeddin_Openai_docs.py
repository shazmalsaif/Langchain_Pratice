from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

docs = [
    "My name is shzamal?",
    "I am learning langchain.",
    "I love programming.",
]

embed = embeddings.embed_documents(docs)

print(str(embed))

