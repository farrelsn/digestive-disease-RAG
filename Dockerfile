# Railway builds this image. Python 3.12 rather than 3.14: torch and its
# dependencies have had wheels here for much longer, so the build doesn't
# fall back to compiling from source.
FROM python:3.12-slim

# Run as a normal user rather than root — good practice, and what most hosts expect.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Install the CPU-only build of torch FIRST. The default wheel on PyPI bundles
# CUDA libraries — several gigabytes that do nothing without a GPU, and enough
# to push this image past Railway's size limit.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies second: this layer is cached and only rebuilds when requirements.txt changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image (about 130 MB), so starting the app
# never waits on a download and the running container needs no network for it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=user . .

# Build the vector database here instead of committing data/chroma_db/ to git.
# It keeps binary files out of the repo, and the database can never be
# out of step with the data/chunks.jsonl it was built from.
RUN python embed.py

# Railway assigns a port at runtime and passes it in as $PORT.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
