# Akira server: the API, the full app at /app, and the embeddable widget at /embed.js.
#   docker build -t akira .
#   docker run -p 8000:8000 -e AKIRA_ORIGINS=https://yoursite.com -e ANTHROPIC_API_KEY=... akira
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY akira akira
COPY web web
COPY release models
ENV AKIRA_DEVICE=none AKIRA_MODELS=models PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-W", "ignore", "-m", "akira.server"]
