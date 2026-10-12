# Rubik's cube web app served by the trained policy network (CPU only).
#   docker build -t rubiks-nn-app .
#   docker run --rm -p 8765:8765 rubiks-nn-app        then open http://localhost:8765/
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Dependencies first, so code changes do not re-download them. torch comes from the PyTorch CPU index: the default
# Linux x86-64 wheel on PyPI pulls the CUDA libraries (gigabytes) that this image never uses. On arm64 the wheels are
# CPU-only anyway, so the same line works on both architectures.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
 && pip install "numpy>=2"

WORKDIR /app
COPY pyproject.toml ./
COPY rubiks ./rubiks
COPY web ./web
COPY models ./models

# An unprivileged user; the files stay owned by root and are made world-readable whatever the checkout's umask was.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && chmod -R a+rX,go-w /app
USER app

# Search settings, read at start-up (docker run -e BEAM=256 ...). CHECKPOINT can point at a mounted file.
ENV BEAM=1024 \
    MAX_DEPTH=40 \
    CHECKPOINT=/app/models/policy-v2.pt

EXPOSE 8765

# The server turns SIGINT into a clean shutdown; as PID 1 it would ignore the default SIGTERM.
STOPSIGNAL SIGINT

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", \
       "import json, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:8765/solver', timeout=4)); raise SystemExit(0 if d.get('backend') == 'policy' else 1)"]

# A tiny shell entrypoint so the environment variables apply; extra arguments to `docker run IMAGE ...` are appended
# (a later flag wins, e.g. `docker run ... rubiks-nn-app --beam 256`).
ENTRYPOINT ["/bin/sh", "-c", \
            "exec python -m rubiks.serve --host 0.0.0.0 --port 8765 --web /app/web --backend policy --checkpoint \"$CHECKPOINT\" --beam \"$BEAM\" --max-depth \"$MAX_DEPTH\" --device cpu --fallback none \"$@\"", \
            "rubiks-nn-app"]
