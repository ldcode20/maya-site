# Maya demo website (Render, or Hugging Face Spaces with PRO). Built from the
# folder scripts/build_space.py assembles: src/, frontend/, models/ (the trained
# bundle), serve.py and requirements.txt.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MAYA_TRUST_PROXY=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home maya

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=maya:maya src ./src
COPY --chown=maya:maya frontend ./frontend
COPY --chown=maya:maya models ./models
COPY --chown=maya:maya serve.py ./

USER maya
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/v1/health', timeout=3)"
# serve.py binds 0.0.0.0:$PORT (Render) or :7860 (Spaces), one worker.
CMD ["python", "serve.py"]
