# Retrieval: Find the best chunks for a user question

import re

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.stem import PorterStemmer

from chunking import load_chunks

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # use the same model as embed.py

# Hybrid search parameters
CANDIDATES = 10
TOP_K = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
RANK_CONSTANT = 1

stemmer = PorterStemmer()

def tokenize(text):
    words = re.findall(r"\w+", text.lower())
    return [stemmer.stem(word) for word in words]

def build_hybrid_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Open vector database created in embed.py
    vector_store = Chroma(
        collection_name="digestive_health",
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
    )
    if not vector_store.get(limit=1)["ids"]:
        raise RuntimeError("data/chroma_db is still empty. Run embed.py first.")
    semantic_search = vector_store.as_retriever(search_kwargs={"k": CANDIDATES})

    # Keyword search: BM25 builds its word index in memory from the same chunks.
    keyword_search = BM25Retriever.from_documents(load_chunks(), preprocess_func=tokenize, k=CANDIDATES)

    # Hybrid search: a chunk ranked high by either search ends up high in the merged list.
    # id_key tells it that the same chunk_id from both searches is the same chunk.
    return EnsembleRetriever(
        retrievers=[semantic_search, keyword_search],
        weights=[SEMANTIC_WEIGHT, KEYWORD_WEIGHT],
        c=RANK_CONSTANT,
        id_key="chunk_id",
    )


def retrieve(retriever, question):
    """Return the TOP_K best chunks for a question."""
    return retriever.invoke(question)[:TOP_K]


# for testing purposes
def show(label, chunks):
    print(f"  {label}:")
    for rank, chunk in enumerate(chunks[:TOP_K], start=1):
        print(f"    {rank}. {chunk.metadata['chunk_id']}")

if __name__ == "__main__":
    hybrid = build_hybrid_retriever()
    semantic_search, keyword_search = hybrid.retrievers

    questions = [
        "What is the difference between GER and GERD?",
        "What foods should I avoid with celiac disease?",
        "How is appendicitis diagnosed?",
    ]

    # Compare the three methods side by side for each question.
    for question in questions:
        print(f"\nQuestion: {question}")
        show("Semantic only", semantic_search.invoke(question))
        show("Keyword only (BM25)", keyword_search.invoke(question))
        show("Hybrid", retrieve(hybrid, question))
