# Digestive Health Explainer

A RAG (Retrieval-Augmented Generation) chatbot that answers digestive health questions using
published NIDDK articles. It searches ~1,000 passages of NIDDK text, hands the best ones to an
LLM, and returns an answer where every statement carries a `[1]` linking to the page it came
from. It never answers from the model's own knowledge — if the sources don't cover something,
it says so.

This explains published health information. It does not diagnose anyone or recommend
treatment.

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
python chunking.py     # articles → chunks.jsonl
python embed.py        # chunks   → chroma_db/

# 4. Start the app
python run.py
```

Step 4 opens [127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The first run of `embed.py` downloads the embedding model (about 130 MB). After that,
everything except the Groq call runs offline on your own machine, with no GPU needed.

### In the terminal instead

```bash
python generate.py
```

Same pipeline, asked and answered at the command line. Press Enter on an empty line to quit.
