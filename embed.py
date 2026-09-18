from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from chunking import load_chunks

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5") 

chunks = load_chunks()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=[chunk.metadata["chunk_id"] for chunk in chunks],
    collection_name="digestive_health",
    persist_directory="chroma_db",
)