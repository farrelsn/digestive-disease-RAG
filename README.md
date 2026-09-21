# Digestive Health Explainer

A Retrieval-Augmented Generation (RAG) chatbot that answers digestive health questions using articles scraped from the official National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) website:

https://www.niddk.nih.gov/

The application combines semantic and keyword search to retrieve relevant passages, then uses a language model to generate answers grounded in those passages.

Live app: https://digestive-disease-rag-production.up.railway.app/

## Tech stack

**Data collection**
* `requests` + `BeautifulSoup` — scrape the NIDDK articles ([health_sources_collections.ipynb](notebook/health_sources_collections.ipynb))

**RAG pipeline** (LangChain)
* `RecursiveCharacterTextSplitter` — split articles into chunks ([chunking.py](chunking.py))
* `BAAI/bge-small-en-v1.5` via HuggingFace embeddings — turn chunks into vectors ([embed.py](embed.py))
* Chroma — vector database, stored in `data/chroma_db/` ([embed.py](embed.py))
* BM25 (`rank_bm25`) + NLTK stemming — keyword search ([retrieve.py](retrieve.py))
* `EnsembleRetriever` — hybrid search, 70% semantic / 30% keyword ([retrieve.py](retrieve.py))
* Groq (`openai/gpt-oss-120b`) — generates the answer from the retrieved chunks ([generate.py](generate.py))

**Web app**
* FastAPI + Uvicorn — Backend ([app.py](app.py))
* HTML, CSS and JavaScript — Frontend ([index.html](index.html))

**Deployment**
* Docker + Railway — the image bakes in the embedding model and builds the vector database at build time ([Dockerfile](Dockerfile), [railway.json](railway.json))


## Running locally

**Prerequisites:** Python 3.10 or newer, and a free [Groq API key](https://console.groq.com/keys).

```bash
# 1. Install
python -m venv .venv && .venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env → set GROQ_API_KEY

# 3. Build the search index
python chunking.py     # articles → data/chunks.jsonl
python embed.py        # chunks   → data/chroma_db/

# 4. Start the app
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in browser.

### In the terminal instead

```bash
python generate.py
```

Same pipeline, asked and answered at the command line. Press Enter on an empty line to quit.
