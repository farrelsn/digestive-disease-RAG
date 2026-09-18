# Creating the chunks.jsonl file to store into vector database

import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "health_documents.json"
OUTPUT_FILE = "chunks.jsonl"

CHUNK_SIZE = 800    # limit the chunk size to 800 characters, so the LLM can read multiple chunks in one prompt
CHUNK_OVERLAP = 120  # keep some overlap between chunks to not lose context


def remove_footnote_numbers(text):
    """Remove citation numbers such as 'have GERD. 1' or 'sphincter. 4,5 They'."""
    return re.sub(r"(?<=[.!?])\s+\d{1,2}(,\d{1,2})*(?=\s|$)", "", text)


def article_text(document):
    """Join all sections of an article into one text, keeping each heading."""
    sections = []
    for section in document["sections"]:
        text = remove_footnote_numbers(section["text"])
        sections.append(f"## {section['heading']}\n{text}")
    return "\n\n".join(sections)


def load_chunks(path=OUTPUT_FILE):
    """Read chunks.jsonl back into LangChain Documents"""
    with open(path, encoding="utf-8") as file:
        return [Document(**json.loads(line)) for line in file]


def main():
    with open(INPUT_FILE, encoding="utf-8") as file:
        documents = json.load(file)

    documents = [doc for doc in documents if not doc["document_id"].endswith("_clinical_trials")]

    # using LangChain Recursive text splitter to help split the article into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Split the document_id into condition and page_type, e.g. "gerd_overview" -> "gerd", "overview"
        condition, page_type = doc["document_id"].rsplit("_", 1)

        for number, piece in enumerate(splitter.split_text(article_text(doc))):
            chunks.append(Document(
                # Start every chunk with the article title, so a passage like
                # "Esophagitis is inflammation..." still tells us it is about GERD.
                page_content=f"{doc['title']}\n\n{piece}",
                metadata={
                    "chunk_id": f"{doc['document_id']}_{number}",
                    "document_id": doc["document_id"],
                    "condition": condition,
                    "page_type": page_type,
                    "title": doc["title"],
                    "source_url": doc["source_url"],
                    "last_reviewed": doc["last_reviewed"] or "",
                },
            ))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for chunk in chunks:
            row = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    word_counts = [len(chunk.page_content.split()) for chunk in chunks]
    print(f"Articles used: {len(documents)}")
    print(f"Chunks saved:  {len(chunks)} -> {OUTPUT_FILE}")
    print(f"Words per chunk: smallest {min(word_counts)}, largest {max(word_counts)}")
    print("Example chunk:\n")
    print(chunks[0].page_content)
    print("-" * 50)
    print(chunks[0].metadata)


if __name__ == "__main__":
    main()
