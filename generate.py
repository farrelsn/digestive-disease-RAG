# Generation: Answer a question using LLM and the retrieved chunks.
import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from retrieve import build_hybrid_retriever, retrieve

# Read GROQ_API_KEY from the .env file
load_dotenv()
if os.getenv("GROQ_API_KEY", "your-key-here") == "your-key-here":
    raise RuntimeError("No Groq API key found. Put your key in the .env file: GROQ_API_KEY=your-key-here")

# The list of current models: https://console.groq.com/docs/models
GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You explain digestive health information to the public, using ONLY the numbered sources below.

Rules:
- Base every statement on the sources. Do not add facts from your own knowledge.
- After each statement, cite its source number in square brackets, like [1] or [2][3].
  Use that exact format. No other citation style, and never add line numbers.
- If the sources don't answer the question, say: "My sources don't cover this." Don't guess.
- Some sources are written for children or infants (their title says so). Unless the question is about a child or infant, use the other sources, and say so if you rely on a child or infant source.
- You explain published information. You don't diagnose anyone or tell them what treatment to choose. When it fits, suggest talking with a doctor.
- If the question describes serious warning signs (for example vomiting blood, black or bloody stool, severe belly pain, signs of dehydration), tell the person to seek medical care right away.
- Use plain, simple language.

Sources:
{sources}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_sources(chunks):
    """Number the chunks so the LLM can cite them as [1], [2], ..."""
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        sources.append(f"[{number}] {chunk.page_content}")
    return "\n\n".join(sources)


def normalize_citations(text):
    """
    Some models fall back to the citation style they were trained on and write things like
    "【1†L7-L11】" instead of "[1]". This function normalizes them to the correct style.
    """
    return re.sub(r"【(\d+)[^】]*】", r"[\1]", text)


def answer(question, retriever, llm):
    chunks = retrieve(retriever, question)
    response = (prompt | llm).invoke({"sources": format_sources(chunks), "question": question})
    return normalize_citations(response.content), chunks


def cited_numbers(text):
    """The source numbers the answer actually used, found by looking for [1], [2][3], ..."""
    return sorted({int(number) for number in re.findall(r"\[(\d+)\]", text)})


def cited_sources(text, chunks):
    """List only the sources the answer cited. Retrieval hands the LLM more chunks than it
    needs, so listing all of them would credit pages the answer never used.

    One page is usually split into several chunks, so the numbers are collected per page
    instead of repeating the same title and link two or three times.
    """
    pages = {}
    for number in cited_numbers(text):
        if not 1 <= number <= len(chunks):
            continue  # the model made up a number that wasn't offered to it
        meta = chunks[number - 1].metadata
        page = pages.setdefault(meta["source_url"], {
            "url": meta["source_url"],
            "title": meta["title"],
            "last_reviewed": meta["last_reviewed"],
            "numbers": [],
        })
        page["numbers"].append(number)
    return list(pages.values())


# testing purposes
def print_sources(text, chunks):
    pages = cited_sources(text, chunks)
    if not pages:
        print("\n(The answer cited no source.)")
        return

    print("\nSources:")
    for page in pages:
        labels = "".join(f"[{number}]" for number in page["numbers"])
        print(f"  {labels} {page['title']} ({page['last_reviewed']})")
        print(f"      {page['url']}")


if __name__ == "__main__":
    retriever = build_hybrid_retriever()
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)  # temperature 0 = stick closely to the sources

    print("\nAsk a question about digestive health (press Enter on an empty line to quit).")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break
        text, chunks = answer(question, retriever, llm)
        print("\n" + text)
        print_sources(text, chunks)
