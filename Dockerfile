# Akira server: the API, the full app at /app, and the embeddable widget at /embed.js.
#   docker build -t akira .
#   docker run -p 8000:8000 -e AKIRA_ORIGINS=https://yoursite.com akira
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY akira akira
COPY web web
COPY release models
# Akira's shared encyclopedia (Simple Wikipedia, ~330 MB index), built at image build time
RUN python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.simple/train/0.parquet', '/tmp/simplewiki.parquet')" \
 && python -c "from akira import knowledge; knowledge.build('/tmp/simplewiki.parquet', 'models/knowledge.db')" \
 && rm /tmp/simplewiki.parquet
ENV AKIRA_DEVICE=none AKIRA_MODELS=models PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-W", "ignore", "-m", "akira.server"]
