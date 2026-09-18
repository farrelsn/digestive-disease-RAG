"""
Run locally:   uvicorn app:app
               then open http://127.0.0.1:8000
Deployed:      Hugging Face Spaces builds the Dockerfile and serves on port 7860.
"""

# FastAPI server for the digestive health question-answering app.
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from generate import GROQ_MODEL, answer, cited_sources
from retrieve import build_hybrid_retriever

# The link is public and every question spends Groq credits, so each visitor
# gets a limited number of questions per hour.
QUESTIONS_PER_HOUR = 20
ask_history = defaultdict(list)  # visitor -> the times they asked

# Build the retriever and LLM once when the server starts.
print("Loading the retriever...")
retriever = build_hybrid_retriever()
llm = ChatGroq(model=GROQ_MODEL, temperature=0)  # temperature 0 = stick closely to the sources

app = FastAPI()


class Question(BaseModel):
    # A real question is short. The cap stops anyone pasting a whole book into the prompt.
    question: str = Field(min_length=1, max_length=500)


def visitor_id(request):
    """Who is asking. Spaces sits behind a proxy, which puts the real address in this header."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or request.client.host


def check_rate_limit(visitor):
    """Allow QUESTIONS_PER_HOUR questions per visitor, counting the last hour only."""
    an_hour_ago = time.time() - 3600
    recent = [asked_at for asked_at in ask_history[visitor] if asked_at > an_hour_ago]

    if len(recent) >= QUESTIONS_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"That's {QUESTIONS_PER_HOUR} questions this hour. Please try again later.",
        )

    recent.append(time.time())
    ask_history[visitor] = recent


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/ask")
def ask(question: Question, request: Request):
    check_rate_limit(visitor_id(request))
    text, chunks = answer(question.question, retriever, llm)
    return {"answer": text, "sources": cited_sources(text, chunks)}
