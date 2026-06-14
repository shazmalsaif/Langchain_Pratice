from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "My name is shzamal?",
    "I am learning langchain.",
    "I love programming.",
]


embed = embeddings.embed_documents(docs)

print(str(embed))
