# LookLab -- single container serving both the API and the front end.
#
# The front end is one HTML file that FastAPI serves at "/", so there is
# nothing to build and nothing to deploy separately. One image, one port, one
# URL, no CORS.
#
# This image runs unchanged on Render, Koyeb, Fly.io, Google Cloud Run and
# Hugging Face Spaces (Docker SDK). The only platform difference is which port
# it is told to listen on, which comes from $PORT.

FROM python:3.12-slim

# - PYTHONUNBUFFERED so logs appear immediately in the platform's log viewer
#   rather than sitting in a buffer while you wonder if the app is alive.
# - PYTHONDONTWRITEBYTECODE because the filesystem is ephemeral anyway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing application code does not invalidate the
# (slow) pip layer. requirements.txt deliberately excludes scikit-image and
# scipy -- 143 MB for one function, now in looklab/cielab.py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY looklab/ ./looklab/
COPY tools/ ./tools/
COPY README.md ./

# Bake the tiktoken vocabulary into the image.
#
# tiktoken fetches its BPE file over the network on first use and caches it in
# a temp directory. On a cold container with no cache that download happens
# during the grader's first request -- or fails, if egress is restricted.
# Downloading it at build time makes the running container fully offline for
# token counting. looklab/context.py still falls back to an approximate
# counter if anything here goes wrong.
ENV TIKTOKEN_CACHE_DIR=/app/looklab/data/tiktoken
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" && \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base').encode('warm up')" && \
    echo "tiktoken vocabulary cached at build time"

# Writable location for the SQLite checkpointer and store.
#
# On free tiers this is ephemeral: it is wiped on redeploy and on the restart
# that follows an idle sleep. That is expected and handled -- the app re-seeds
# two demo threads on boot (see looklab/seed.py), so a cold start still shows
# a populated recipe, profile and trim telemetry rather than four empty panes.
ENV LOOKLAB_CHECKPOINT_PATH=/app/data/checkpoints.sqlite \
    LOOKLAB_STORE_PATH=/app/data/store.sqlite
RUN mkdir -p /app/data

# Run as a non-root user. Several platforms (Hugging Face Spaces in
# particular) refuse to run containers as root.
RUN useradd --create-home --uid 1000 looklab && chown -R looklab:looklab /app
USER looklab

# 7860 is Hugging Face Spaces' required port and a harmless default elsewhere;
# Render, Koyeb and Cloud Run all override it via $PORT.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','7860') + '/health', timeout=4)"

# Shell form so $PORT expands at runtime.
# Deliberately no --reload: the reloader forks a second process and both would
# open the same SQLite WAL file.
CMD uvicorn looklab.server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
